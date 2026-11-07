"""
this file generated the answers to all the test cases in eval data and stores it into the json file for 
evaluation
"""


from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from dotenv import load_dotenv

logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

load_dotenv(override=True)

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(root_dir, "rag_system"))
os.chdir(root_dir)
from langchain_core.output_parsers import StrOutputParser
from rag_retrieval_pipeline import llm, prompt, retriever

# creating dedicated chain in answer generation for testing in order to avoide recurrent calls 
qa_chain = prompt | llm | StrOutputParser()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(
    current_dir, f"rag_generation_output_similarity_{timestamp}.json"
)

EVAL_DATA = [
    # ---------------- Apollo 11 ----------------
    {
        "question": (
            "Who were the three crew members of Apollo 11 and what were their"
            " roles?"
        ),
        "ground_truth": (
            "Neil A. Armstrong was the Commander, Michael Collins was the"
            " Command Module Pilot, and Edwin E. Aldrin, Jr. was the Lunar"
            " Module Pilot."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": "When and from where was Apollo 11 launched?",
        "ground_truth": (
            "Apollo 11 was launched from Kennedy Space Center, Florida, at"
            " 8:32:00 a.m. EST on July 16, 1969."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "How many midcourse corrections were needed during the Apollo 11"
            " translunar coast, and when was it performed?"
        ),
        "ground_truth": (
            "Only one midcourse correction was required, performed at"
            " approximately 27 hours ground elapsed time."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "How did the actual Apollo 11 landing point compare with the"
            " planned landing point?"
        ),
        "ground_truth": (
            'The actual landing point in the Sea of Tranquility was latitude 0'
            ' deg 41\'15" N, longitude 23 deg 26\' E, compared with the'
            " planned point of latitude 0 deg 43'53\" N, longitude 23 deg 38'51\""
            " E."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "Which computer alarms occurred during Apollo 11 powered descent,"
            " and how did the crew respond when the automatic descent was"
            " heading toward a boulder field?"
        ),
        "ground_truth": (
            "1201 and 1202 alarms occurred during powered descent. Because the"
            " automatic descent would have ended in a boulder field near a large"
            " sharp-rimmed crater, the crew assumed manual control, extended"
            " the range, and used the rate-of-descent throttle control mode"
            " (program P66)."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "Which scientific experiments did the Apollo 11 crew deploy on the"
            " lunar surface, and how much lunar material did they collect?"
        ),
        "ground_truth": (
            "The crew deployed a solar wind experiment, a passive seismic"
            " experiment, and a laser retroreflector, and collected"
            " approximately 47 pounds of lunar surface material."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "What quarantine procedures were followed after Apollo 11"
            " splashdown, and when were the crew released?"
        ),
        "ground_truth": (
            "After recovery by helicopter to the U.S.S. Hornet, the crew and"
            " lunar samples were placed in the Mobile Quarantine Facility, then"
            " transported to the Lunar Receiving Laboratory in Houston. The"
            " crew and spacecraft were released from quarantine on August 10,"
            " 1969."
        ),
        "is_out_of_scope": False,
    },
    # ---------------- Apollo 13 ----------------
    {
        "question": (
            "Who chaired the Apollo 13 Review Board and when was the final"
            " report transmitted?"
        ),
        "ground_truth": (
            "Edgar M. Cortright chaired the Apollo 13 Review Board, and the"
            " final report was transmitted to NASA Administrator Thomas O."
            " Paine on June 15, 1970."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "What primary mechanical event caused the failure of Oxygen Tank"
            " No. 2 in the Apollo 13 Service Module?"
        ),
        "ground_truth": (
            "A combustion process within oxygen tank no. 2, involving ignition"
            " of Teflon wire insulation, caused rapid pressure and temperature"
            " rise. This led to failure of the tank closure or conduit,"
            " pressurization of the tank dome and rupture of its disc,"
            " releasing oxygen and combustion products into bay 4 of the"
            " service module, whose panel was missing after the accident."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "At what mission time did the Apollo 13 incident begin, and what"
            " did the data show happened in oxygen tank no. 2?"
        ),
        "ground_truth": (
            "The incident began at about 55 hours 52 minutes ground elapsed"
            " time. Oxygen tank no. 2 pressure and temperature rose, pointing"
            " to a combustion process, with pressure peaking at about 1008 psia."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "Why was John Swigert on the Apollo 13 prime crew instead of Thomas"
            " Mattingly?"
        ),
        "ground_truth": (
            "Backup Lunar Module Pilot Charles Duke contracted rubella in the"
            " week before launch. Tests showed Lovell and Haise were immune but"
            " Command Module Pilot Thomas Mattingly was not, so backup Command"
            " Module Pilot John Swigert was substituted after two days of"
            " simulator training."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": "What were the primary mission objectives of Apollo 13?",
        "ground_truth": (
            "To perform selenological inspection, survey, and sampling of the"
            " Fra Mauro Formation; deploy and activate an Apollo Lunar Surface"
            " Experiments Package (ALSEP); develop man's capability to work in"
            " the lunar environment; and photograph candidate exploration"
            " sites."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "Why was the lunar module used as a 'lifeboat' on Apollo 13 and by"
            " when was it activated?"
        ),
        "ground_truth": (
            "The oxygen tank no. 1 leak could not be stopped, so the LM had to"
            " serve as a lifeboat. By 58:40 ground elapsed time the LM was"
            " activated, inertial guidance reference was transferred to the LM"
            " guidance system, and the CSM systems were turned off."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "Which consumables became critical on Apollo 13 after the"
            " accident, and what improved the situation?"
        ),
        "ground_truth": (
            "Water, electrical power (batteries), oxygen, and lithium hydroxide"
            " canisters for carbon dioxide removal were marginal. Powering"
            " down the LM to conserve electric power and reduce heat and water"
            " use greatly improved the situation."
        ),
        "is_out_of_scope": False,
    },
    # ---------------- Challenger ----------------
    {
        "question": (
            "When did Challenger Mission 51-L launch and how long did the"
            " flight last?"
        ),
        "ground_truth": (
            "Mission 51-L launched at 11:38 a.m. EST on January 28, 1986, and"
            " ended 73 seconds later in an explosive burn of hydrogen and"
            " oxygen propellants that destroyed the External Tank and exposed"
            " the Orbiter to severe aerodynamic loads."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": "Who were the seven crew members lost on Challenger?",
        "ground_truth": (
            "Francis R. (Dick) Scobee (Commander), Michael J. Smith (Pilot),"
            " Ellison S. Onizuka, Judith A. Resnik, and Ronald E. McNair"
            " (Mission Specialists), and S. Christa McAuliffe and Gregory B."
            " Jarvis (Payload Specialists)."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "What was the ambient temperature at the Challenger launch and how"
            " did it compare with earlier launches?"
        ),
        "ground_truth": (
            "The ambient temperature at launch was 36 degrees Fahrenheit, which"
            " was 15 degrees lower than the next coldest previous launch."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "Where did the leak that caused the Challenger accident originate,"
            " and what features of the joint were involved?"
        ),
        "ground_truth": (
            "The combustion gas leak originated in the right Solid Rocket Motor"
            " aft field joint. The motor case sections are joined by"
            " tang-and-clevis joints held by 177 steel pins, sealed with"
            " primary and secondary O-rings and protected by zinc chromate"
            " putty."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "Which Morton Thiokol engineers objected to the Challenger launch"
            " and what was Thiokol's initial recommendation?"
        ),
        "ground_truth": (
            "Roger Boisjoly and Arnold Thompson voiced strong objections."
            " Thiokol's initial recommendation, from Bob Lund, was not to"
            " launch until the O-ring temperature reached 53 degrees"
            " Fahrenheit, the lowest temperature of any previous flight."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "What role did Lawrence Mulloy and the launch constraint waivers"
            " play in the Challenger accident?"
        ),
        "ground_truth": (
            "Lawrence Mulloy, the Solid Rocket Booster Project Manager at"
            " Marshall, regularly waived the launch constraint related to the"
            " O-ring problem, including six consecutive waivers before 51-L."
            " Neither the constraint, its reason, nor the waivers were known"
            " to Level I or Level II management."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "What happened during Morton Thiokol's internal caucus before"
            " reversing its recommendation not to launch?"
        ),
        "ground_truth": (
            "About ten engineers participated along with managers including"
            " Mason, Kilminster, Wiggins and Lund. Boisjoly and Thompson"
            " objected strongly, but a senior manager asked Bob Lund to take"
            " off his engineering hat and put on his management hat, after"
            " which management formulated the points on which to base a"
            " decision to launch."
        ),
        "is_out_of_scope": False,
    },
    # ---------------- Cross-document ----------------
    {
        "question": (
            "Who appears in both Apollo 11 and Rogers Commission in what role?"
        ),
        "ground_truth": (
            "Neil A. Armstrong served as the Commander of Apollo 11 (the first"
            " manned lunar landing) and later served as the Vice Chairman of"
            " the Presidential Commission on the Space Shuttle Challenger"
            " Accident (the Rogers Commission)."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "How did temperature play a role in the Apollo 13 and Challenger"
            " accident?"
        ),
        "ground_truth": (
            "In Challenger, cold ambient temperatures at launch (36°F) caused"
            " severe O-ring resiliency degradation and loss of sealing"
            " capability in the right Solid Rocket Motor aft field joint,"
            " resulting in hot gas blow-by. In Apollo 13, abnormally high"
            " temperatures during pre-flight detanking caused thermostatic"
            " switches to fail and weld closed, allowing heater temperatures to"
            " reach approximately 1000°F and bake off wire insulation, which"
            " eventually triggered the oxygen tank combustion."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "Which of Apollo 13 and Challenger involved a last-minute crew"
            " change?"
        ),
        "ground_truth": (
            "Apollo 13 involved a last-minute crew change when backup Command"
            " Module Pilot John L. Swigert, Jr., replaced prime CMP Thomas K."
            " Mattingly III just days prior to launch due to rubella exposure."
            " Challenger had no last-minute crew change."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "Did the Apollo 13 incident occur before or after Apollo 11?"
        ),
        "ground_truth": (
            "After. Apollo 11 took place in July 1969, while the Apollo 13"
            " mission and oxygen tank explosion occurred nine months later in"
            " April 1970."
        ),
        "is_out_of_scope": False,
    },
    {
        "question": (
            "Describe the use of lithium hydroxide in Apollo 11 and Apollo 13?"
        ),
        "ground_truth": (
            "In Apollo 11, lithium hydroxide filters in the environmental"
            " control system were used during cabin cleanup to capture abrasive"
            " airborne lunar dust after surface EVAs. In Apollo 13, lithium"
            " hydroxide canisters were critical consumables used to scrub"
            " carbon dioxide from the atmosphere; after the accident, ground"
            " controllers devised a procedure using spacesuit hoses, cue cards,"
            " and plastic bags to adapt square Command Module LiOH canisters"
            " for use in the Lunar Module's cylindrical system."
        ),
        "is_out_of_scope": False,
    },
    # ---------------- Out-of-scope / false-premise ----------------
    {
        "question": "What was the cause of the Columbia accident in 2003?",
        "ground_truth": (
            "This is not covered in the provided documents (Apollo 11, Apollo"
            " 13, and Challenger reports). The system should state that it"
            " cannot answer from the available context."
        ),
        "is_out_of_scope": True,
    },
    {
        "question": "What is the current status of NASA's Artemis program?",
        "ground_truth": (
            "This is not covered in the provided documents. The system should"
            " state that the documents do not contain information about the"
            " Artemis program."
        ),
        "is_out_of_scope": True,
    },
    {
        "question": "Why did the Apollo 11 descent engine fail during landing?",
        "ground_truth": (
            "False premise. The Apollo 11 report does not describe a descent"
            " engine failure; it describes 1201/1202 computer alarms and a"
            " manual takeover to avoid a boulder field. The system should"
            " correct the premise."
        ),
        "is_out_of_scope": True,
    },
    {
        "question": (
            "What did the Apollo 13 Review Board conclude about the Challenger"
            " explosion?"
        ),
        "ground_truth": (
            "False premise. The Apollo 13 Review Board reported in 1970, years"
            " before the 1986 Challenger accident, and could not have drawn"
            " conclusions about it. The system should say this cannot be"
            " answered."
        ),
        "is_out_of_scope": True,
    },
]


OUTPUT_FILE = os.path.join(current_dir, "rag_generation_output2.json")


def generate_and_save():
  """Generates answers and retrieves contexts for the evaluation data"""
  
  records = []
  total = len(EVAL_DATA)
  print(f"Generating answers and retrieving contexts for {total} queries...")

  for idx, item in enumerate(EVAL_DATA, 1):
    q = item["question"]
    gt = item["ground_truth"]
    is_oos = item.get("is_out_of_scope", False)

    # 1. RETRIEVE ONCE
    retrieved_docs = retriever.invoke(q)

    # Handle both Document lists (from standard/hybrid retrievers) and raw strings
    if (
        isinstance(retrieved_docs, list)
        and retrieved_docs
        and hasattr(retrieved_docs[0], "page_content")
    ):
      contexts = [doc.page_content for doc in retrieved_docs]
    elif isinstance(retrieved_docs, list):
      contexts = [str(doc) for doc in retrieved_docs]
    else:
      contexts = [str(retrieved_docs)]

    context_str = (
        "\n\n".join(contexts)
        if contexts
        else "I cannot find relevant information in the indexed mission"
        " documents."
    )

    # 2. GENERATE USING THE IDENTICAL RETRIEVED CONTEXT
    answer_text = qa_chain.invoke({"context": context_str, "question": q})

    # 3. SAVE FOR RAGAS
    records.append({
        "question": q,
        "contexts": contexts,  # Exact chunks evaluated
        "answer": answer_text,
        "ground_truth": gt,
        "is_out_of_scope": is_oos,
    })

    print(f"[{idx}/{total}] Done: {q[:45]}...")
    time.sleep(5)  # Pace free-tier RPM limits

  with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

  print(f"\nGeneration complete. Data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
  generate_and_save()