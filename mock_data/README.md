# TRUST Platform - Mock Data

Test data for development and validation without needing live Cerner connection.

## Files

### scribe_notes/

| File | Description | Use Case |
|------|-------------|----------|
| `sample_note_1.json` | Clean AI scribe note | Test normal workflow |
| `sample_note_2_hallucinated.json` | Note WITH intentional hallucinations | Test detection algorithms |

## Hallucination Types in sample_note_2

The second sample contains these **intentional errors** for testing:

1. **Fabricated Medication**: Warfarin listed but patient denied taking it
2. **Fabricated Allergy**: Sulfa anaphylaxis added (only penicillin mentioned)
3. **Fabricated Exam Finding**: Leg edema noted but transcript says "no swelling"
4. **Fabricated Recommendation**: IABP standby never discussed

## JSON Structure

Each note contains:
- `patient`: Demographics
- `encounter`: Visit details  
- `ai_scribe_output`: What the AI generated
- `source_transcript`: Original conversation (ground truth)
- `hallucination_flags`: (optional) Known issues for validation

## Adding New Test Cases

When creating new mock data:
1. Write a realistic transcript first
2. Generate a plausible AI note
3. Intentionally add specific error types
4. Document errors in `hallucination_flags`
