"""
lexicon.py — The Mechatronics Keyword Brain

This file is the single source of truth for every technical term the system
knows about. Think of it as a dictionary of "words that matter" for Mechatronics.

When the parser scans a CV, it checks every text block against this list.
If a word from this list appears in the CV, we flag it as a recognized skill.

All keywords are stored in LOWERCASE — matching is always done case-insensitively,
so "MATLAB", "matlab", and "Matlab" all count as the same hit.

We use a frozenset (an immutable, unordered set) because:
- Sets give us O(1) membership checks — fast even with 240+ keywords
- frozenset can't be accidentally modified at runtime
- No duplicates possible by design

The keywords come from two sources:
1. The full-form terms in mechatronics_keyword_weights.json (e.g. "solidworks", "ros2")
2. Short-form abbreviations people actually write on CVs (e.g. "pid", "plc", "fpga")
   that aren't captured by the longer forms above
"""

MECHATRONICS_KEYWORDS: frozenset = frozenset({

    # ─────────────────────────────────────────────────────────────
    # INDUSTRIAL AUTOMATION — PLCs, SCADA, HMI, Motor Control
    # ─────────────────────────────────────────────────────────────

    # Siemens PLC family — the most common PLCs in Egyptian and regional industry
    "siemens s7-1200",
    "siemens s7-1500",
    "siemens s7-300",
    "siemens s7-400",
    "s7-1200",       # short form — engineers write this without the "siemens" prefix
    "s7-1500",
    "s7-300",
    "s7-400",
    "tia portal",    # Siemens programming environment
    "step 7",        # older Siemens software, still widely used
    "wincc",         # Siemens SCADA / HMI software

    # Schneider Electric PLC family
    "schneider electric",
    "zelio logic",
    "ecostruxure machine expert",
    "modicon m221",
    "modicon m241",
    "modicon m251",
    "unity pro",
    "vijeo designer",

    # Other popular PLCs
    "classic plc ladder",   # ladder logic programming language
    "abb plc",
    "mitsubishi fx series",
    "omron cp1e",
    "omron cx-programmer",
    "delta plc",
    "wplsoft",
    "ispsoft",
    "allen-bradley micrologix",
    "allen-bradley compactlogix",
    "rslogix 500",
    "studio 5000",

    # Short-form PLC / IEC 61131-3 programming language names
    # (people write these all the time without spelling out the full name)
    "plc",
    "function block diagram (fbd)",
    "structured text (st)",
    "instruction list (il)",
    "sequential function chart (sfc)",

    # Motor drives and electrical control hardware
    "relay logic",
    "contactor wiring",
    "motor control center (mcc)",
    "vfd (variable frequency drive)",
    "danfoss vfd",
    "siemens sinamics",
    "abb acs drives",
    "induction motors",
    "servo motors",
    "stepper motors",
    "pneumatics",
    "festo fluidsim",
    "electro-pneumatics",
    "hydraulics",
    "proportional valves",

    # Control and sensing
    "pid controller tuning",
    "pid",           # PID is the #1 most-written control term on Mechatronics CVs
    "sensor calibration",
    "proximity sensors",
    "photoelectric sensors",
    "limit switches",
    "encoder feedback",

    # Industrial software and communication
    "factory io",
    "scada",
    "hmi",
    "hmi design",
    "ignition scada",
    "modbus rtu",
    "modbus tcp/ip",
    "modbus",        # short form used by itself
    "profinet",
    "profibus",
    "ethernet/ip",
    "ethercat",      # not in the JSON but mentioned in the plan — very common in automation
    "opc ua",
    "opc-ua",        # hyphen variant people write


    # ─────────────────────────────────────────────────────────────
    # ROBOTICS — ROS, motion planning, simulation, robot hardware
    # ─────────────────────────────────────────────────────────────

    # ROS (Robot Operating System) — the de-facto standard for robotics software
    "ros2 humble",
    "ros (robot operating system)",
    "ros2",
    "ros",           # many engineers just write "ROS" without the full name
    "gazebo simulator",
    "rviz",
    "moveit!",
    "urdf (unified robot description format)",
    "tf (transformations)",

    # Motion and kinematics theory
    "kinematics",
    "forward kinematics",
    "inverse kinematics",
    "jacobian matrix",
    "robot dynamics",

    # Path planning algorithms — you'll see these in robotics project descriptions
    "path planning",
    "a* algorithm",
    "dijkstra",
    "rrt (rapidly-exploring random tree)",

    # SLAM and perception
    "slam (simultaneous localization and mapping)",
    "slam",          # people write "SLAM" without expanding the acronym
    "lidar processing",
    "point cloud processing",
    "pcl (point cloud library)",
    "odometer calibration",

    # Mobile robots and drive types
    "mobile robots",
    "agv (automated guided vehicle)",
    "amr (autonomous mobile robot)",
    "differential drive",
    "omnidirectional drive",

    # Computer vision — often overlaps with Robotics and Automotive
    "machine vision",
    "opencv",
    "yolo (you only look once)",
    "image processing",
    "camera calibration",

    # Industrial robot brands and their programming languages
    "kuka robots",
    "kuka krl",
    "abb irc5",
    "rapid programming",
    "fanuc robots",
    "karel",
    "yaskawa motoman",
    "universal robots (ur)",
    "urscript",
    "cobots (collaborative robots)",

    # Robot simulation and offline programming tools
    "robodk",
    "robotstudio",
    "webots",
    "coppelisim",
    "isaac sim",
    "robot control systems",
    "impedance control",
    "admittance control",


    # ─────────────────────────────────────────────────────────────
    # EMBEDDED SYSTEMS — microcontrollers, protocols, firmware tools
    # ─────────────────────────────────────────────────────────────

    # PCB and circuit design tools
    "altium designer",
    "proteus",
    "proteus isis",
    "eagle cad",
    "kicad",
    "pcb design",

    # Popular microcontroller families and IDEs
    "arduino",
    "avr atmega32",
    "microcontrollers",
    "microcontroller",   # singular form — also common
    "pic microcontrollers",
    "arm cortex-m",
    "arm",               # generic ARM mention
    "cortex-m4",         # specific core used in STM32F4 series
    "cortex-m3",
    "stm32",
    "stm32cubemx",       # STMicroelectronics configuration tool
    "stm32cubeide",      # STMicroelectronics full IDE
    "keil uvision",
    "iar embedded workbench",
    "microchip studio",
    "atmel studio",

    # Simulation tools
    "multisim",
    "tinkercad",
    "fritzing",

    # Operating systems and programming paradigms for embedded
    "freertos",
    "rtos",              # generic RTOS — people write this when they mean FreeRTOS, Zephyr, etc.
    "bare-metal programming",
    "embedded c",
    "embedded systems",  # the field itself — appears in section headers and skill lists

    # Low-level hardware programming concepts
    "bit manipulation",
    "interrupt handling",
    "watchdog timer",
    "systick timer",

    # Communication protocols — the "language" chips use to talk to each other
    "spi protocol",
    "spi",       # short form
    "i2c protocol",
    "i2c",       # short form
    "uart protocol",
    "uart",      # short form
    "can bus",
    "can",       # short form — careful, but common enough to include
    "adc (analog-to-digital converter)",
    "adc",
    "dac (digital-to-analog converter)",
    "dac",
    "pwm (pulse width modulation)",
    "pwm",
    "dma (direct memory access)",
    "dma",

    # IoT and wireless hardware
    "esp32",
    "esp8266",
    "iot (internet of things)",
    "mqtt",
    "node-red",

    # Debugging and measurement instruments
    "oscilloscope operation",
    "logic analyzer",
    "hardware debugging",


    # ─────────────────────────────────────────────────────────────
    # AUTOMOTIVE — ECU, CAN, AUTOSAR, safety standards, HIL/SIL
    # ─────────────────────────────────────────────────────────────

    # AUTOSAR — the software architecture framework used by every major OEM
    "autosar",
    "autosar classic",
    "autosar adaptive",

    # Vector tools — the go-to toolchain for automotive CAN/LIN analysis
    "canalyzer",
    "canoe",
    "vector tools",
    "capl scripting",

    # Automotive communication protocols
    "uds (unified diagnostic services)",
    "lin bus",
    "lin",       # short form
    "flexray",
    "automotive ethernet",
    "obd-ii",

    # Engine and powertrain control
    "ecu (engine control unit)",
    "ecu",       # appears standalone very often
    "ecu flashing",
    "inca (etas)",
    "canape",

    # Safety standards — critical for automotive job applications
    "misra c",
    "iso 26262",
    "functional safety",

    # Model-Based Design (MBD) — used heavily in automotive ECU development
    "matlab",
    "simulink",
    "stateflow",
    "model-based design (mbd)",
    "mil (model-in-the-loop)",
    "sil (software-in-the-loop)",
    "hil (hardware-in-the-loop)",
    "dspace",
    "carmaker",
    "prescan",

    # Electric vehicles and ADAS
    "electric vehicles (ev) powertrain",
    "battery management system (bms)",
    "bms",       # short form
    "inverter control",
    "adas (advanced driver assistance systems)",
    "adas",      # short form
    "lane keeping assist (lka)",
    "adaptive cruise control (acc)",
    "sensor fusion",
    "kalman filter",


    # ─────────────────────────────────────────────────────────────
    # MECHANICAL DESIGN — CAD, FEA, manufacturing
    # ─────────────────────────────────────────────────────────────

    # The big CAD packages — SolidWorks is by far the most common in Egypt
    "solidworks",
    "solidworks sheet metal",
    "solidworks weldments",
    "autocad",
    "classic autocad",
    "catia",
    "fusion 360",
    "inventor",
    "freecad",

    # Simulation and analysis
    "ansys",
    "ansys fluent",
    "ansys mechanical",
    "fea (finite element analysis)",
    "fea",
    "cfd (computational fluid dynamics)",
    "cfd",

    # Manufacturing processes
    "3d printing",
    "cura slicer",
    "g-code",
    "cnc machining",
    "cam (computer-aided manufacturing)",
    "mastercam",
    "sheet metal design",
    "plastics design",
    "injection molding basics",

    # Mechanical engineering fundamentals — these appear in Skills sections
    "gd&t (geometric dimensioning and tolerancing)",
    "tolerance stack-up analysis",
    "statics",
    "dynamics",
    "thermodynamics",
    "fluid mechanics",
    "mechanics of materials",
    "material selection",
    "machine design",
    "gears design",
    "shaft design",
    "bearings selection",


    # ─────────────────────────────────────────────────────────────
    # TECHNICAL MANAGEMENT — project tools, soft skills, standards
    # ─────────────────────────────────────────────────────────────

    # Programming languages that appear across multiple niches
    "python",
    "c++",
    "c",
    "fpga",      # not in JSON — Field Programmable Gate Array, from plan.md
    "vhdl",      # hardware description language for FPGAs
    "verilog",   # alternative HDL
    "labview",   # LabVIEW from National Instruments — in plan.md

    # Project and process management
    "project management",
    "pmp basics",
    "agile methodology",
    "scrum framework",
    "jira",
    "trello",

    # Engineering process frameworks
    "systems engineering",
    "v-model",
    "requirements engineering",
    "doors",
    "risk management",
    "fmea (failure mode and effects analysis)",
    "fmea",

    # Soft skills and communication — these show up in management-track CVs
    "leadership",
    "technical writing",
    "problem solving",
    "root cause analysis",
    "5 whys",
    "fishbone diagram",
    "team collaboration",
    "cross-functional communication",
    "time management",
    "presentation skills",
    "excel for engineers",
    "data analysis",
    "cost estimation",

    # Version control
    "version control",
    "git",
    "github",
    "gitlab",
    "bitbucket",
})
