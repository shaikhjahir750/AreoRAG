"""
this file generated the answers to all the test cases in eval data and stores it into the json file for 
evaluation
"""


import json
import os
import sys
import time
from dotenv import load_dotenv
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
load_dotenv(override=True)

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(root_dir, "rag_system"))
os.chdir(root_dir)

from rag_retrieval_pipeline import retriever, rag_chain

EVAL_DATA = [
    # --- Apollo 13 Review (Cortright Report) [7 Questions] ---
    {
        "question": "What primary mechanical event caused the failure of Oxygen Tank No. 2 in the Apollo 13 Service Module?",
        "ground_truth": "Combustion or electrical arcing within Oxygen Tank No. 2 ignited Teflon insulation on the wiring, causing rapid pressure buildup, structural failure of the tank, and blow-off of the outer SM panel."
    },
    {
        "question": "Why did the heater thermostatic switches fail during the pre-flight detanking of Apollo 13?",
        "ground_truth": "Heater switch contacts were rated for 28V DC but subjected to 65V DC ground power at KSC, causing contacts to weld shut and overheat the tank wiring."
    },
    {
        "question": "What role did the Lunar Module 'Aquarius' play immediately following the oxygen tank rupture?",
        "ground_truth": "Aquarius served as a lifeboat, providing life support, electrical power, and propulsion via the DPS while the Command Module was powered down."
    },
    {
        "question": "What peak temperature did the internal heater tube in Oxygen Tank No. 2 reach during ground detanking?",
        "ground_truth": "Internal heater tube temperatures reached approximately 1000°F (540°C) due to continuous current through welded switches."
    },
    {
        "question": "How did the handling incident at North American Rockwell contribute to the Apollo 13 tank failure?",
        "ground_truth": "The oxygen shelf assembly dropped two inches during removal from Apollo 10, loosening internal fill line plumbing."
    },
    {
        "question": "What redesign recommendations were made by the Cortright Commission for the CSM oxygen system?",
        "ground_truth": "Remove internal heater and fan motors, isolate wiring from high-pressure oxygen, add a third auxiliary oxygen tank, and install an emergency battery."
    },
    {
        "question": "How did the Apollo 13 crew resolve toxic CO2 buildup inside the Lunar Module?",
        "ground_truth": "They adapted square CM lithium hydroxide canisters to the round LM sockets using plastic bags, cardboard, and tape (the mailbox)."
    },

    # --- Challenger STS 51-L (Rogers Commission Report) [7 Questions] ---
    {
        "question": "What was the direct physical cause of the Challenger STS 51-L launch failure?",
        "ground_truth": "Failure of the primary and secondary O-ring seals in the right SRB aft field joint allowed combustion gases to escape and impinge on the External Tank."
    },
    {
        "question": "How did cold ambient temperatures impair SRB O-ring functionality?",
        "ground_truth": "Cold temperatures increased Viton rubber hardness and delayed seal resiliency, preventing proper sealing of the joint during ignition pressure."
    },
    {
        "question": "What objection did Morton Thiokol engineers raise the evening prior to STS 51-L launch?",
        "ground_truth": "Engineers recommended not launching below 53°F due to historical correlation between low temperature and O-ring erosion."
    },
    {
        "question": "How did NASA management treat prior O-ring erosion issues before STS 51-L?",
        "ground_truth": "NASA management normalized the deviance, treating recurring O-ring erosion as an acceptable flight risk rather than a Criticality 1 hazard."
    },
    {
        "question": "What demonstration did Richard Feynman conduct during the Rogers Commission hearings?",
        "ground_truth": "He submerged a compressed O-ring material sample in ice water to demonstrate its loss of elasticity at near-freezing temperatures."
    },
    {
        "question": "What modifications were made to the SRB field joint following the Rogers Commission?",
        "ground_truth": "The joint was redesigned with a capture feature metal lip to prevent rotation and installed electric heater bands to maintain seal temperatures at or above 75°F."
    },
    {
        "question": "What role did zinc chromate putty play in the SRB field joint leakage?",
        "ground_truth": "Displacement of putty during joint assembly created air pockets/blowholes that focused high-pressure combustion jets directly onto the O-rings."
    },

    # --- Apollo 11 Mission Report [7 Questions] ---
    {
        "question": "What caused the 1201 and 1202 program alarms on the Apollo 11 Lunar Module guidance computer?",
        "ground_truth": "Executive overflow alarms caused by the Apollo Guidance Computer being overloaded with radar cycle calculations."
    },
    {
        "question": "What hardware switch position triggered the AGC computer overload during lunar descent?",
        "ground_truth": "The rendezvous radar switch was set to AUTO rather than SLEW, feeding continuous unnecessary pulse interrupts into the processor."
    },
    {
        "question": "Why did Neil Armstrong switch to semi-automatic manual control (P66) during descent?",
        "ground_truth": "The automatic guidance path was directing the spacecraft toward a boulder-filled crater field around West Crater."
    },
    {
        "question": "How much descent propellant remained when Apollo 11 touched down?",
        "ground_truth": "Approximately 25 to 45 seconds of usable descent engine fuel remained before the abort limit."
    },
    {
        "question": "What scientific instruments comprised the Apollo 11 EASEP package?",
        "ground_truth": "The Passive Seismic Experiment Package (PSEP) and the Laser Ranging Retro-Reflector (LRRR)."
    },
    {
        "question": "How did Aldrin activate the broken ascent engine circuit breaker before lunar liftoff?",
        "ground_truth": "He pushed the broken circuit breaker switch in using a felt-tip marker pen."
    },
    {
        "question": "What quarantine protocol was applied to the Apollo 11 crew after splashdown?",
        "ground_truth": "The crew wore biological isolation garments, were scrubbed with Betadine, entered a Mobile Quarantine Facility, and completed 21 days of isolation."
    },

    # --- Cross-Document Reasoning & Synthesis [6 Questions] ---
    {
        "question": "Compare the operational role of the Lunar Module in Apollo 11 versus Apollo 13.",
        "ground_truth": "In Apollo 11, the LM was strictly an exploration craft for two crew members to descend to the Moon. In Apollo 13, it served as an emergency lifeboat for three astronauts across four days."
    },
    {
        "question": "Contrast the pre-launch systemic failures of Apollo 13 and Challenger STS 51-L.",
        "ground_truth": "Apollo 13 resulted from unverified electrical modifications and ground tank damage. Challenger resulted from management launching outside verified temperature limits despite contractor warnings."
    },
    {
        "question": "How did onboard telemetry warnings differ between Apollo 11 and Apollo 13?",
        "ground_truth": "Apollo 11 had AGC cycle duty overflow alarms (1201/1202) while vehicle systems functioned normally. Apollo 13 triggered master alarms from sudden bus voltage and tank pressure losses."
    },
    {
        "question": "How did hardware redesigns after Apollo 13 and Challenger address the root causes of seal and tank integrity?",
        "ground_truth": "Apollo 13 eliminated internal wiring, fans, and heaters from high-pressure oxygen environments, while Challenger added a mechanical capture lip and joint heaters to preserve O-ring seal resiliency."
    },
    {
        "question": "Compare how manual improvisations by crew or engineers overcame mechanical failures in Apollo 11 versus Apollo 13.",
        "ground_truth": "Apollo 11 used a felt-tip marker to engage a broken ascent engine circuit breaker, while Apollo 13 improvised a CO2 scrubber adapter using cardboard, plastic bags, and tape."
    },
    {
        "question": "How did ground testing decisions prior to launch directly compromise flight safety in both Apollo 13 and STS 51-L?",
        "ground_truth": "In Apollo 13, running high-voltage ground detanking welded the heater switches and scorched internal wiring. In Challenger, NASA disregarded prior cold-soak launch data and test erosion trends despite contractor warnings."
    },

    # --- Out-of-Domain Negative Controls (Faithfulness Checks) [3 Questions] ---
    {
        "question": "What caused the thermal protection tile damage on Space Shuttle Columbia STS-107?",
        "ground_truth": "The provided documents contain no information regarding Space Shuttle Columbia or mission STS-107."
    },
    {
        "question": "What were the launch date, crew, and landing site of the Apollo 12 mission?",
        "ground_truth": "The provided documents do not contain information regarding the Apollo 12 mission flight details."
    },
    {
        "question": "Who was the Soviet mission commander during the Apollo-Soyuz Test Project?",
        "ground_truth": "The provided context contains no records regarding the Soviet Union or the Apollo-Soyuz Test Project."
    }
]


OUTPUT_FILE = os.path.join(current_dir, "rag_generation_output.json")


def generate_and_save():
    records = []
    total = len(EVAL_DATA)
    print(f"Generating answers and retrieving contexts for {total} queries...")

    for idx, item in enumerate(EVAL_DATA, 1):
        q = item["question"]
        gt = item["ground_truth"]

        
        retrieved_docs = retriever.invoke(q)
        contexts = [doc.page_content for doc in retrieved_docs]
        answer = rag_chain.invoke(q)

        
        if hasattr(answer, "content"):
            answer_text = answer.content
        elif isinstance(answer, dict):
            answer_text = answer.get("answer") or answer.get("output") or str(answer)
        else:
            answer_text = str(answer)

        records.append({
            "question": q,
            "contexts": contexts,
            "answer": answer_text,
            "ground_truth": gt
        })

        print(f"[{idx}/{total}] Done: {q[:45]}...")
        time.sleep(5)  

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nGeneration complete. Data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_and_save()