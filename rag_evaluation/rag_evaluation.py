"""
This script is to evaluate the rag system that is build in earlier iterations of the project
we are using the LLM as judge technique to evaluate the rag system 
for this we need to use the ragas library and we are using the groq model to evalute the rag 
in order to evalute retrieval we are prefering metrices as context_precision and context_recall
for generation we are umetrics like faithfulness,answer_relevancys
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv(override=True)

# Compatibility shim for ragas with current langchain-community
import langchain_google_vertexai
sys.modules.setdefault("langchain_community.chat_models.vertexai", langchain_google_vertexai)

# Ensure rag_system and project root are accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(root_dir, "rag_system"))
os.chdir(root_dir)

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from rag_retrieval_pipeline import retriever, rag_chain

# 1. Evaluator Setup (Gemini models for high-speed judging)
evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.0)
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
)

# Benchmark test queries generated in relevance to the indexed documents
EVAL_DATA = [
    # --- Apollo 13 Review (Cortright Report) ---
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
        "question": "What was the mission elapsed time (GET) when Jack Swigert reported the initial anomaly?",
        "ground_truth": "The telemetry drop and initial report occurred at approximately 55 hours, 55 minutes GET."
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
        "question": "What caused the pressure decay in Oxygen Tank No. 1 following the explosion of Tank No. 2?",
        "ground_truth": "The blast and debris damaged the connecting line or manifold valve to Tank No. 1, causing a gradual leak."
    },
    {
        "question": "How was the free-return trajectory re-established after the Apollo 13 explosion?",
        "ground_truth": "The crew executed a burn using the Lunar Module Descent Propulsion System (DPS) at approximately 61 hours GET."
    },
    {
        "question": "Why were Command Module RCS propellant and batteries preserved until re-entry?",
        "ground_truth": "They were strictly reserved for atmospheric re-entry because the CM lacked alternative power and attitude control once separated from the LM and SM."
    },
    {
        "question": "How did the Apollo 13 crew resolve toxic CO2 buildup inside the Lunar Module?",
        "ground_truth": "They adapted square CM lithium hydroxide canisters to the round LM sockets using plastic bags, cardboard, and tape (the mailbox)."
    },
    {
        "question": "What was the splashdown location and recovery ship for the Apollo 13 Command Module?",
        "ground_truth": "Odyssey splashed down in the South Pacific Ocean and was recovered by the USS Iwo Jima."
    },
    {
        "question": "What did photographs of the Apollo 13 Service Module show following jettison?",
        "ground_truth": "The entire Sector 4 outer panel was missing, exposing internal structural destruction and ruptured tanks."
    },

    # --- Challenger STS 51-L (Rogers Commission Report) ---
    {
        "question": "What was the direct physical cause of the Challenger STS 51-L launch failure?",
        "ground_truth": "Failure of the primary and secondary O-ring seals in the right SRB aft field joint allowed combustion gases to escape and impinge on the External Tank."
    },
    {
        "question": "What ambient temperature was recorded at Launch Pad 39B on the morning of Challenger's launch?",
        "ground_truth": "Ambient temperature was approximately 36°F (2.2°C), significantly lower than any prior shuttle flight."
    },
    {
        "question": "How did cold ambient temperatures impair SRB O-ring functionality?",
        "ground_truth": "Cold temperatures increased Viton rubber hardness and delayed seal resiliency, preventing proper sealing of the joint during ignition pressure."
    },
    {
        "question": "What visual evidence of joint leakage occurred immediately after Challenger liftoff?",
        "ground_truth": "Puffs of black smoke emerged from the right SRB aft field joint between 0.678 and 2.733 seconds after SRB ignition."
    },
    {
        "question": "What aerodynamic phenomenon intensified the structural failure of Challenger between 56 and 62 seconds?",
        "ground_truth": "The vehicle encountered extreme lateral wind shear forces that induced massive dynamic loads."
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
        "question": "What was the exact mission elapsed time of Challenger's vehicle breakup?",
        "ground_truth": "Challenger broke apart at approximately 73 seconds (T+73.1) at an altitude of ~46,000 feet."
    },
    {
        "question": "What flaw did the Rogers Commission identify in NASA's launch safety hierarchy?",
        "ground_truth": "Safety personnel were understaffed and critical launch-critical contractor objections failed to reach Level I/II decision-makers."
    },
    {
        "question": "What role did zinc chromate putty play in the SRB field joint leakage?",
        "ground_truth": "Displacement of putty during joint assembly created air pockets/blowholes that focused high-pressure combustion jets directly onto the O-rings."
    },
    {
        "question": "What secondary payload was aboard Challenger STS 51-L alongside TDRS-B?",
        "ground_truth": "The Spartan-Halley astronomical research spacecraft."
    },

    # --- Apollo 11 Mission Report ---
    {
        "question": "What caused the 1201 and 1202 program alarms on the Apollo 11 Lunar Module guidance computer?",
        "ground_truth": "Executive overflow alarms caused by the Apollo Guidance Computer being overloaded with radar cycle calculations."
    },
    {
        "question": "What hardware switch position triggered the AGC computer overload during lunar descent?",
        "ground_truth": "The rendezvous radar switch was set to AUTO rather than SLEW, feeding continuous unnecessary pulse interrupts into the processor."
    },
    {
        "question": "What were the exact landing coordinates and site of the Apollo 11 Lunar Module?",
        "ground_truth": "Eagle landed at Mare Tranquillitatis (Sea of Tranquility), approximately 0°40'27\" N latitude and 23°28'23\" E longitude."
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
        "question": "What was the total duration of the Apollo 11 extravehicular activity (EVA)?",
        "ground_truth": "The lunar surface extravehicular activity lasted approximately 2 hours, 31 minutes, and 40 seconds."
    },
    {
        "question": "What total mass of lunar samples was returned by Apollo 11?",
        "ground_truth": "The crew collected and returned approximately 21.55 kg (47.5 pounds) of lunar rocks and soil."
    },
    {
        "question": "How did Aldrin activate the broken ascent engine circuit breaker before lunar liftoff?",
        "ground_truth": "He pushed the broken circuit breaker switch in using a felt-tip marker pen."
    },
    {
        "question": "What launch vehicle configuration injected Apollo 11 into translunar trajectory?",
        "ground_truth": "The three-stage Saturn V rocket (SA-506), comprising S-IC, S-II, and S-IVB stages."
    },
    {
        "question": "What quarantine protocol was applied to the Apollo 11 crew after splashdown?",
        "ground_truth": "The crew wore biological isolation garments, were scrubbed with Betadine, entered a Mobile Quarantine Facility, and completed 21 days of isolation."
    },
    {
        "question": "What was the callsign of the Apollo 11 Command Module and its splashdown area?",
        "ground_truth": "The Command Module was named Columbia and splashed down in the North Pacific Ocean southwest of Hawaii."
    },

    # --- Cross-Document Reasoning ---
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

    # --- Out-of-Domain Negative Controls (Faithfulness checks) ---
    {
        "question": "What caused the thermal protection tile damage on Space Shuttle Columbia STS-107?",
        "ground_truth": "The provided documents contain no information regarding Space Shuttle Columbia or mission STS-107."
    },
    {
        "question": "What were the launch date, crew, and landing site of the Apollo 12 mission?",
        "ground_truth": "The provided documents do not contain information regarding the Apollo 12 mission flight details."
    },
    {
        "question": "What was the cost of replacing the Hubble Space Telescope primary mirror?",
        "ground_truth": "The provided documents do not contain information concerning the Hubble Space Telescope."
    },
    {
        "question": "Who was the Soviet mission commander during the Apollo-Soyuz Test Project?",
        "ground_truth": "The provided context contains no records regarding the Soviet Union or the Apollo-Soyuz Test Project."
    }
]

if __name__ == "__main__":
    questions = [item["question"] for item in EVAL_DATA]
    ground_truths = [item["ground_truth"] for item in EVAL_DATA]
    contexts = []
    answers = []

    print(f"Running evaluation against {len(questions)} queries...")

    for idx, q in enumerate(questions, 1):
        retrieved_docs = retriever.invoke(q)
        contexts.append([doc.page_content for doc in retrieved_docs])
        answers.append(rag_chain.invoke(q))
        print(f"[{idx}/{len(questions)}] Done: {q[:45]}...")
        time.sleep(5)  # Respect Google AI Studio free-tier rate limit (15 RPM)

    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths
    })

    run_config = RunConfig(
    timeout=120,    
    max_workers=2,  
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    df = results.to_pandas()
    if "user_input" in df.columns and "question" not in df.columns:
        df = df.rename(columns={
            "user_input": "question",
            "retrieved_contexts": "contexts",
            "response": "answer",
            "reference": "ground_truth"
        })

    display_cols = [c for c in ["question", "context_precision", "context_recall", "faithfulness", "answer_relevancy"] if c in df.columns]
    print(df[display_cols])
    output_csv = os.path.join(current_dir, "rag_evaluation_report.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nEvaluation complete. Report saved to: {output_csv}")
