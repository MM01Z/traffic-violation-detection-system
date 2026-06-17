src/
│
├─ main.py # Main inference script (run first)
├─ post_nohelmet.py # Post-processing script for No Helmet violation (run second)
├─ violations.py # Violation-specific detection and save functions (functions only)
├─ utils.py # Utility functions (crop, containment check) (functions only)
├─ db_config.py # Database credentials template

models/ 
|
|___# YOLO model weights (custom and default)

├─ data/
│ ├─ videos/ # Input video goes here
│ └─ output/ # Output video with bounding boxes
│
├─ violations/ # Raw detection crops (created by scripts) (temporary, images removed after running post_nohelmet.py)
│ ├─ nohelmet/
│ └─ plates/
│
├─ processed/ # Processed crops after OCR/DB logging
│ └─ nohelmet/
│ 	├─ rider/
│ 	└─ plates/
│
├─ requirementsgpu.txt # Python dependencies
└─ db_schema #mysql code
└─ README.md # This file

python version: 3.11.9

requirements.txt for me
requirementsgpu.txt for taha