from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import random
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# IN-MEMORY DATABASE (Replaces hardcoded JS variables)
# ---------------------------------------------------------
DB = {
    "batches": [],       # Stores metadata for the Batches table
    "rows": [],          # Stores call results
    "transcripts": {},   # Stores dialogue
    "signals": {}        # Stores AI explainability signals
}

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
class CallTranscript(BaseModel):
    id: str
    language: str = "English"
    dialect: str = "Standard"
    call_stage: str = "Unknown"
    transcript: str

class BatchPayload(BaseModel):
    batch_name: str
    calls: List[CallTranscript]

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------

@app.get("/api/state")
async def get_state():
    """Returns the entire current state of the database to populate the frontend on load."""
    return DB

@app.post("/api/classify")
async def classify_batch(payload: BatchPayload):
    """Processes a new JSON upload, runs 'AI', and saves it to the database."""
    time.sleep(1) # Simulate processing delay
    
    batch_conf_total = 0
    new_rows = []
    
    L2_CONTACTED = ["PTP – Full", "PTP – Partial", "Hardship – Job Loss", "Refusal – Explicit"]
    L2_NOT_CONTACTED = ["Not Contacted – No Answer", "Not Contacted – Switched Off"]

    for call in payload.calls:
        # 1. Simulate AI logic
        is_contacted = random.choice([True, True, False])
        l1 = "Contacted" if is_contacted else "Not Contacted"
        l2 = random.choice(L2_CONTACTED) if is_contacted else random.choice(L2_NOT_CONTACTED)
        conf = random.uniform(0.60, 0.99)
        batch_conf_total += conf
        
        # 2. Build Row Data
        row = {
            "id": call.id,
            "batch": payload.batch_name,
            "l1": l1,
            "l2": l2,
            "l3": ["Simulated Tag"] if is_contacted else [],
            "lang": call.language,
            "dialect": call.dialect,
            "conf": conf,
            "l1c": min(conf + 0.05, 0.99),
            "l2c": conf,
            "l3c": conf - 0.05 if is_contacted else None
        }
        new_rows.append(row)
        DB["rows"].insert(0, row) # Add to top of DB
        
        # 3. Build Transcript Data
        formatted_transcript = []
        for line in call.transcript.split("\n"):
            if ":" in line:
                speaker, text = line.split(":", 1)
                formatted_transcript.append([speaker.strip().upper(), text.strip()])
            else:
                formatted_transcript.append(["UNKNOWN", line])
        DB["transcripts"][call.id] = formatted_transcript
        
        # 4. Build Signals Data
        DB["signals"][call.id] = [
            ["+", l1, "Detected connection sequence"],
            ["~" if conf < 0.75 else "+", l2, "Model inference based on context"]
        ]

    # 5. Save Batch Metadata
    avg_conf = batch_conf_total / len(payload.calls) if payload.calls else 0
    new_batch = {
        "name": payload.batch_name,
        "calls": len(payload.calls),
        "status": "Done",
        "time": datetime.now().strftime("%I:%M %p"),
        "avg_conf": avg_conf
    }
    DB["batches"].insert(0, new_batch)

    # Return only what was just created so the frontend can update smoothly
    return {
        "message": "Success",
        "batch": new_batch,
        "rows": new_rows
    }