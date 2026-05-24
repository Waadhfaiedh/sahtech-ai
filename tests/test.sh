#!/bin/bash
PATIENT_TOKEN=$(python3 -c "import jwt; print(jwt.encode({'role': 'patient', 'sub': 'test'}, 'testing-secret-not-for-production', algorithm='HS256'))")
SPECIALIST_TOKEN=$(python3 -c "import jwt; print(jwt.encode({'role': 'specialist', 'sub': 'test'}, 'testing-secret-not-for-production', algorithm='HS256'))")

ROLE=${1:-patient}
MESSAGE=${2:-"j'ai mal à l'épaule depuis 3 jours"}

if [ "$ROLE" = "specialist" ]; then
  TOKEN=$SPECIALIST_TOKEN
else
  TOKEN=$PATIENT_TOKEN
fi

echo "=== Role: $ROLE | Question: $MESSAGE ==="
curl -s -X POST http://localhost:8001/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"$MESSAGE\"}" | python3 -m json.tool