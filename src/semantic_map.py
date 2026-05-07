"""
semantic_map.py — Static Entity-to-Cluster Mapping

This file maps every lowercase entity string in ENTITY_TAXONOMY to one or more
of the six Mechatronics domain cluster names. The mapping is static and
pre-computed — no runtime ML inference happens here.

How this was built:
  The mappings were derived by reasoning about what domain each tool, hardware,
  or standard belongs to in real Mechatronics practice. For example:
  - "s7-1200" is a Siemens PLC → industrial_automation (and nothing else)
  - "autosar" is a software framework for automotive ECUs → automotive, AND
    it also sits deep in embedded_systems territory → both clusters apply
  - "matlab" is used heavily in embedded system design AND mechanical simulation
    → embedded_systems + mechanical_design

Design rules (from research.md R-004):
  1. Every entity in ENTITY_TAXONOMY should have an entry here.
  2. Entities not in this map at runtime → automatically get ["unclassified"].
  3. Multi-cluster assignment is normal and expected.
  4. All keys must be lowercase (they're matched against lowercased entity strings).

The six valid cluster names are:
  "industrial_automation", "robotics", "embedded_systems",
  "automotive", "mechanical_design", "technical_management"

Spec reference: specs/002-semantic-ner-features/data-model.md (SemanticCluster)
Research reference: specs/002-semantic-ner-features/research.md R-004
"""

# The complete static semantic map. Keys are lowercase entity strings.
# Values are lists of one or more cluster names the entity belongs to.
SEMANTIC_MAP: dict[str, list[str]] = {

    # ── PLC HARDWARE → industrial_automation ────────────────────────────────
    # PLCs are the core hardware of industrial automation systems. Some also
    # appear in automotive (e.g., S7 PLCs are used in automotive assembly lines)
    # but their primary home is industrial_automation.
    "s7-1200":                  ["industrial_automation"],
    "s7-1500":                  ["industrial_automation"],
    "s7-300":                   ["industrial_automation"],
    "s7-400":                   ["industrial_automation"],
    "siemens s7-1200":          ["industrial_automation"],
    "siemens s7-1500":          ["industrial_automation"],
    "siemens s7-300":           ["industrial_automation"],
    "compactlogix":             ["industrial_automation"],
    "controllogix":             ["industrial_automation"],
    "allen-bradley compactlogix": ["industrial_automation"],
    "modicon m340":             ["industrial_automation"],
    "modicon m221":             ["industrial_automation"],
    "modicon m241":             ["industrial_automation"],
    "modicon m251":             ["industrial_automation"],
    "omron cp1e":               ["industrial_automation"],
    "delta plc":                ["industrial_automation"],
    "mitsubishi fx series":     ["industrial_automation"],
    "tia portal":               ["industrial_automation"],
    "step 7":                   ["industrial_automation"],

    # ── CAD SOFTWARE → mechanical_design ────────────────────────────────────
    # CAD packages are the backbone of mechanical design work. They appear
    # under mechanical_design. SolidWorks and CATIA also appear later under
    # mechanical_tool with the same cluster assignment — that's fine because
    # the enricher deduplicates by (term, type, section) triple, not by cluster.
    "solidworks":               ["mechanical_design"],
    "catia":                    ["mechanical_design"],
    "autocad":                  ["mechanical_design"],
    "fusion 360":               ["mechanical_design"],
    "creo":                     ["mechanical_design"],
    "nx":                       ["mechanical_design"],
    "inventor":                 ["mechanical_design"],
    "freecad":                  ["mechanical_design"],
    "solidworks sheet metal":   ["mechanical_design"],
    "solidworks weldments":     ["mechanical_design"],

    # ── MICROCONTROLLER → embedded_systems ──────────────────────────────────
    # Microcontrollers are the heart of embedded systems. All MCU families
    # map directly to embedded_systems. Some also have automotive relevance
    # (e.g., ARM Cortex-M is used in automotive ECUs) so we double-map those.
    "cortex-m4":                ["embedded_systems", "automotive"],
    "cortex-m3":                ["embedded_systems"],
    "arm cortex-m":             ["embedded_systems", "automotive"],
    "stm32":                    ["embedded_systems"],
    "arduino":                  ["embedded_systems"],
    "esp32":                    ["embedded_systems"],
    "esp8266":                  ["embedded_systems"],
    "raspberry pi":             ["embedded_systems", "robotics"],
    "atmega":                   ["embedded_systems"],
    "avr atmega32":             ["embedded_systems"],
    "pic microcontrollers":     ["embedded_systems"],
    "microcontroller":          ["embedded_systems"],
    "microcontrollers":         ["embedded_systems"],

    # ── COMMUNICATION PROTOCOL → varies by protocol origin ──────────────────
    # Protocols have clear primary homes:
    #   - CAN Bus, PROFINET, Modbus → industrial_automation + embedded_systems
    #   - EtherCAT, PROFIBUS → industrial_automation
    #   - I2C, SPI, UART → embedded_systems (low-level chip protocols)
    #   - OPC-UA → industrial_automation (IIoT standard)
    #   - MQTT → embedded_systems (IoT)
    #   - FlexRay, LIN, CAN FD → automotive (vehicle bus networks)
    "can bus":                  ["automotive", "embedded_systems"],
    "profinet":                 ["industrial_automation"],
    "profibus":                 ["industrial_automation"],
    "ethercat":                 ["industrial_automation"],
    "modbus":                   ["industrial_automation"],
    "modbus rtu":               ["industrial_automation"],
    "modbus tcp/ip":            ["industrial_automation"],
    "i2c":                      ["embedded_systems"],
    "spi":                      ["embedded_systems"],
    "uart":                     ["embedded_systems"],
    "opc-ua":                   ["industrial_automation"],
    "opc ua":                   ["industrial_automation"],
    "ethernet/ip":              ["industrial_automation"],
    "flexray":                  ["automotive"],
    "lin bus":                  ["automotive"],
    "automotive ethernet":      ["automotive"],
    "mqtt":                     ["embedded_systems"],

    # ── PROGRAMMING LANGUAGE → depends on language domain ───────────────────
    # Most languages are cross-domain, but we assign them based on the primary
    # Mechatronics use case:
    #   - MATLAB → embedded_systems + mechanical_design (dual-use tool)
    #   - Python → robotics + embedded_systems (heavily used in both)
    #   - C++ → embedded_systems + automotive (common in ECU/firmware work)
    #   - Embedded C → embedded_systems (by definition)
    #   - Ladder Logic, ST, FBD → industrial_automation (IEC 61131-3 languages)
    #   - VHDL, Verilog → embedded_systems (FPGA / digital design)
    #   - LabVIEW → industrial_automation (National Instruments ecosystem)
    #   - CAPL → automotive (Vector tool scripting)
    #   - RAPID, KRL, URScript → robotics (robot-manufacturer languages)
    "matlab":                   ["embedded_systems", "mechanical_design"],
    "python":                   ["robotics", "embedded_systems"],
    "c++":                      ["embedded_systems", "automotive"],
    "embedded c":               ["embedded_systems"],
    "ladder logic":             ["industrial_automation"],
    "structured text":          ["industrial_automation"],
    "function block diagram":   ["industrial_automation"],
    "vhdl":                     ["embedded_systems"],
    "verilog":                  ["embedded_systems"],
    "labview":                  ["industrial_automation"],
    "capl scripting":           ["automotive"],
    "urscript":                 ["robotics"],
    "rapid programming":        ["robotics"],
    "kuka krl":                 ["robotics"],

    # ── SIMULATION TOOL → depends on what domain uses the tool ──────────────
    # MATLAB → already mapped above
    # Simulink → embedded_systems + automotive (Model-Based Design is huge in auto)
    # ANSYS → mechanical_design (FEA/CFD is purely mechanical engineering)
    # Gazebo → robotics (the standard ROS robot simulator)
    # Adams, COMSOL → mechanical_design
    # dSPACE, Stateflow → automotive (MBD toolchain for ECU development)
    # Prescan, CarMaker → automotive (ADAS/AV simulation)
    # RoboDK, RobotStudio, Webots → robotics
    "simulink":                 ["embedded_systems", "automotive"],
    "ansys":                    ["mechanical_design"],
    "labview":                  ["industrial_automation"],     # already above but repeated for simulation
    "gazebo simulator":         ["robotics"],
    "adams":                    ["mechanical_design"],
    "comsol":                   ["mechanical_design"],
    "dspace":                   ["automotive"],
    "stateflow":                ["automotive"],
    "model-based design":       ["automotive", "embedded_systems"],
    "prescan":                  ["automotive"],
    "carmaker":                 ["automotive"],
    "robodk":                   ["robotics"],
    "robotstudio":              ["robotics"],
    "webots":                   ["robotics"],
    "coppelisim":               ["robotics"],

    # ── ROBOTIC FRAMEWORK → robotics ────────────────────────────────────────
    # All robotics frameworks map to robotics. OpenCV also has industrial
    # machine vision uses but its primary home in a Mechatronics context is robotics.
    "ros":                      ["robotics"],
    "ros2":                     ["robotics"],
    "ros2 humble":              ["robotics"],
    "ros (robot operating system)": ["robotics"],
    "moveit":                   ["robotics"],
    "moveit!":                  ["robotics"],
    "opencv":                   ["robotics"],
    "pcl (point cloud library)": ["robotics"],
    "rviz":                     ["robotics"],
    "slam":                     ["robotics"],
    "opencv":                   ["robotics"],

    # ── AUTOMOTIVE STANDARD → automotive (+ embedded_systems where applicable) ──
    # These are purely automotive standards — their sole purpose is to govern
    # automotive ECU development processes and safety.
    # AUTOSAR and ISO 26262 also touch embedded_systems because ECU firmware
    # is the subject matter.
    "autosar":                  ["embedded_systems", "automotive"],
    "autosar classic":          ["embedded_systems", "automotive"],
    "autosar adaptive":         ["embedded_systems", "automotive"],
    "misra c":                  ["automotive", "embedded_systems"],
    "iso 26262":                ["automotive"],
    "aspice":                   ["automotive"],
    "can fd":                   ["automotive"],
    "functional safety":        ["automotive"],
    "uds (unified diagnostic services)": ["automotive"],

    # ── MECHANICAL TOOL → mechanical_design ─────────────────────────────────
    # Mechanical analysis and manufacturing tools all belong to mechanical_design.
    # ANSYS has the same mapping as above — consistent regardless of type.
    "ansys mechanical":         ["mechanical_design"],
    "ansys fluent":             ["mechanical_design"],
    "fea":                      ["mechanical_design"],
    "fea (finite element analysis)": ["mechanical_design"],
    "cfd":                      ["mechanical_design"],
    "cfd (computational fluid dynamics)": ["mechanical_design"],
    "gd&t (geometric dimensioning and tolerancing)": ["mechanical_design"],
    "cnc machining":            ["mechanical_design"],
    "mastercam":                ["mechanical_design"],
    "3d printing":              ["mechanical_design"],

    # ── MANAGEMENT METHODOLOGY → technical_management ───────────────────────
    # Project management frameworks and engineering process methodologies all
    # map to technical_management — this is the cleanest 1:1 mapping in the
    # entire taxonomy. Systems Engineering and V-Model have some overlap with
    # automotive (both are used heavily in ASPICE / ISO 26262 projects) so
    # we double-map those.
    "agile":                    ["technical_management"],
    "scrum":                    ["technical_management"],
    "kanban":                   ["technical_management"],
    "six sigma":                ["technical_management"],
    "pmp":                      ["technical_management"],
    "fmea":                     ["technical_management"],
    "fmea (failure mode and effects analysis)": ["technical_management"],
    "v-model":                  ["technical_management", "automotive"],
    "systems engineering":      ["technical_management"],
    "risk management":          ["technical_management"],
    "requirements engineering": ["technical_management"],
}
