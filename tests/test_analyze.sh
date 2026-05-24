#!/bin/bash
ROLE=${1:-patient}
TOKEN=$(python3 -c "import jwt; print(jwt.encode({'role': '$ROLE', 'sub': 'test'}, 'testing-secret-not-for-production', algorithm='HS256'))")

echo "=== Movement Report | Role: $ROLE ==="
curl -s -X POST http://localhost:8001/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_name": "Jean-Dominique Morel",
    "body_part": "épaule",
    "session_date": "2026-05-22",
    "affected_side": "left",
    "metrics": {
      "flexion":          {"left":142,"right":168,"delta_left":12,"delta_right":2,"unit":"°","reference":180},
      "extension":        {"left":38,"right":55,"delta_left":5,"delta_right":1,"unit":"°","reference":60},
      "abduction":        {"left":110,"right":165,"delta_left":8,"delta_right":0,"unit":"°","reference":180},
      "adduction":        {"left":28,"right":40,"delta_left":3,"delta_right":0,"unit":"°","reference":45},
      "rotation_externe": {"left":45,"right":80,"delta_left":-2,"delta_right":1,"unit":"°","reference":90},
      "rotation_interne": {"left":55,"right":70,"delta_left":4,"delta_right":0,"unit":"°","reference":70}
    },
    "movement_quality": {
      "controle_moteur":"EXCELLENT",
      "compensations":"MODÉRÉ",
      "dyskinesie_scapulaire":"SUIVI"
    },
    "symmetry_score": 89,
    "recovery_score": 76,
    "mobility_pct": 82,
    "pain_pct": 14,
    "pain_scale_eva_change": -2,
    "language": "fr"
  }' | python3 -m json.tool