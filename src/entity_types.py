"""
entity_types.py — The 10-Type Mechatronics NER Taxonomy

This file is the single source of truth for every named entity type the
Phase 2 NER system recognizes, and the specific entity strings that belong
to each type.

How to read this file:
  ENTITY_TAXONOMY is a dict where:
    - Each KEY is a canonical entity type label (one of exactly 10 types)
    - Each VALUE is a list of lowercase entity strings that belong to that type

  The NER scanner in enricher.py iterates over this structure:
  for every entity string in any type's list, it searches the CV text for
  that string (case-insensitive, whole-word). On a match, it emits one
  NamedEntity record per type the matching string belongs to.

Multi-type entities (entities that appear in more than one type's list):
  - "matlab"     → programming_language AND simulation_tool
  - "solidworks" → cad_software AND mechanical_tool
  - "catia"      → cad_software AND mechanical_tool
  - "ansys"      → simulation_tool AND mechanical_tool
  These appear in multiple lists intentionally — the spec (FR-003) explicitly
  requires multi-type entity support.

All entity strings are lowercase. Matching in enricher.py is case-insensitive,
so "SolidWorks", "solidworks", and "SOLIDWORKS" on a CV will all match "solidworks".

Spec reference: specs/002-semantic-ner-features/data-model.md (EntityType table)
Research reference: specs/002-semantic-ner-features/research.md R-003
"""

# The complete Phase 2 entity taxonomy. Exactly 10 type keys — immutable in Phase 2.
# If you add a new type in a future phase, update the feature vector schema too
# (VECTOR_INDEX_ORDER in feature_vector.py), because the vector size is derived
# from the number of types × sections.
ENTITY_TAXONOMY: dict[str, list[str]] = {

    # ── PLC HARDWARE ─────────────────────────────────────────────────────────
    # Programmable Logic Controller hardware models — the physical boxes that
    # run industrial automation programs. We include both the full "siemens s7-1200"
    # form and the abbreviated "s7-1200" form because engineers write both.
    "plc_hardware": [
        "s7-1200",
        "s7-1500",
        "s7-300",
        "s7-400",
        "siemens s7-1200",
        "siemens s7-1500",
        "siemens s7-300",
        "compactlogix",
        "controllogix",
        "allen-bradley compactlogix",
        "modicon m340",
        "modicon m221",
        "modicon m241",
        "modicon m251",
        "omron cp1e",
        "delta plc",
        "mitsubishi fx series",
        "tia portal",       # Siemens programming environment — closely tied to the hardware
        "step 7",
    ],

    # ── CAD SOFTWARE ─────────────────────────────────────────────────────────
    # Computer-Aided Design packages used to model physical parts and assemblies.
    # SolidWorks and CATIA also appear in mechanical_tool because they are both
    # design tools and analysis tools — the spec explicitly supports this.
    "cad_software": [
        "solidworks",       # also in mechanical_tool (multi-type)
        "catia",            # also in mechanical_tool (multi-type)
        "autocad",
        "fusion 360",
        "creo",
        "nx",
        "inventor",
        "freecad",
        "solidworks sheet metal",
        "solidworks weldments",
    ],

    # ── MICROCONTROLLER ──────────────────────────────────────────────────────
    # Microcontroller families and embedded processor cores. These are the chips
    # that firmware engineers program at a low level — not to be confused with
    # full computers like Raspberry Pi (which also appears here because engineers
    # frequently use it for embedded-adjacent projects and list it as such).
    "microcontroller": [
        "cortex-m4",
        "cortex-m3",
        "arm cortex-m",
        "stm32",
        "arduino",
        "esp32",
        "esp8266",
        "raspberry pi",
        "atmega",
        "avr atmega32",
        "pic microcontrollers",
        "microcontroller",
        "microcontrollers",
    ],

    # ── COMMUNICATION PROTOCOL ───────────────────────────────────────────────
    # Industrial and embedded communication protocols — the standards that define
    # how devices on a network exchange data. We include both long-form
    # ("can bus") and short-form ("can") because CVs use both styles.
    "communication_protocol": [
        "can bus",
        "profinet",
        "profibus",
        "ethercat",
        "modbus",
        "modbus rtu",
        "modbus tcp/ip",
        "i2c",
        "spi",
        "uart",
        "opc-ua",
        "opc ua",
        "ethernet/ip",
        "flexray",
        "lin bus",
        "automotive ethernet",
        "mqtt",
    ],

    # ── PROGRAMMING LANGUAGE ─────────────────────────────────────────────────
    # Programming and scripting languages used in Mechatronics roles.
    # MATLAB appears here AND in simulation_tool because engineers use it
    # both as a programming environment and as a simulation platform.
    "programming_language": [
        "matlab",           # also in simulation_tool (multi-type)
        "python",
        "c++",
        "embedded c",
        "ladder logic",
        "structured text",
        "function block diagram",
        "vhdl",
        "verilog",
        "labview",
        "capl scripting",
        "urscript",
        "rapid programming",
        "kuka krl",
    ],

    # ── SIMULATION TOOL ──────────────────────────────────────────────────────
    # Engineering simulation and modeling environments. Some tools (MATLAB, ANSYS)
    # serve double duty as both simulation tools and analysis tools, hence
    # their presence in multiple type lists.
    "simulation_tool": [
        "matlab",           # also in programming_language (multi-type)
        "simulink",
        "ansys",            # also in mechanical_tool (multi-type)
        "labview",
        "gazebo simulator",
        "adams",
        "comsol",
        "dspace",
        "stateflow",
        "model-based design",
        "prescan",
        "carmaker",
        "robodk",
        "robotstudio",
        "webots",
        "coppelisim",
    ],

    # ── ROBOTIC FRAMEWORK ────────────────────────────────────────────────────
    # Robotics software frameworks, middleware, and libraries.
    # ROS and ROS2 are by far the most common in Mechatronics CVs.
    "robotic_framework": [
        "ros",
        "ros2",
        "ros2 humble",
        "ros (robot operating system)",
        "moveit",
        "moveit!",
        "opencv",
        "pcl (point cloud library)",
        "rviz",
        "slam",
        "openCV",
    ],

    # ── AUTOMOTIVE STANDARD ──────────────────────────────────────────────────
    # Automotive engineering standards, methodologies, and architecture frameworks.
    # These are the certifications and standards that gate entry to automotive ECU
    # development roles.
    "automotive_standard": [
        "autosar",
        "autosar classic",
        "autosar adaptive",
        "misra c",
        "iso 26262",
        "aspice",
        "can fd",
        "functional safety",
        "uds (unified diagnostic services)",
    ],

    # ── MECHANICAL TOOL ──────────────────────────────────────────────────────
    # Mechanical engineering analysis and design tools. SolidWorks and CATIA
    # appear here in addition to cad_software — they are design tools (CAD)
    # and they are also used for mechanical analysis (FEA, weldments, sheet metal).
    # ANSYS appears here and in simulation_tool for the same dual-purpose reason.
    "mechanical_tool": [
        "solidworks",       # also in cad_software (multi-type)
        "catia",            # also in cad_software (multi-type)
        "ansys",            # also in simulation_tool (multi-type)
        "ansys mechanical",
        "ansys fluent",
        "fea",
        "fea (finite element analysis)",
        "cfd",
        "cfd (computational fluid dynamics)",
        "gd&t (geometric dimensioning and tolerancing)",
        "cnc machining",
        "mastercam",
        "3d printing",
    ],

    # ── MANAGEMENT METHODOLOGY ───────────────────────────────────────────────
    # Project management and process improvement methodologies.
    # These appear on CVs of engineers targeting team lead or project manager roles
    # in the Mechatronics space.
    "management_methodology": [
        "agile",
        "scrum",
        "kanban",
        "six sigma",
        "pmp",
        "fmea",
        "fmea (failure mode and effects analysis)",
        "v-model",
        "systems engineering",
        "risk management",
        "requirements engineering",
    ],
}
