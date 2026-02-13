# Sign Convention Fix - Implementation Summary

## Overview
Fixed fundamental inconsistencies in how signs are handled across different modes in the Kt optimizer. The implementation now uses consistent, physically meaningful sign conventions throughout.

## Changes Made

### 1. Core Physics Changes

**Old Convention (Inconsistent):**
- Mixed signed and magnitude-based physics
- LINKED mode: Used signed forces in LP but flipped signs arbitrarily in display (Kt- = -Kt+)
- INDIVIDUAL mode: Used magnitude forces in LP but signed forces in validation
- Negative Kt values were allowed, which is physically nonsensical

**New Convention (Consistent):**
- **All modes:** Kt ≥ 0 (non-negative, enforced by LP bounds)
- **All modes:** Forces are signed (F ∈ ℝ)
- **All modes:** Stresses are signed (σ ∈ ℝ)
- **Physical relationship:** σ = Kt × F (stress sign comes from force sign)

### 2. Files Modified

#### `kt_optimizer/models.py` (lines 100-133)
- **Fixed:** `expand_kt_to_canonical()` function
- **Change:** Removed sign flip for LINKED mode (line 132)
  - Old: `values_out.append(-raw)` for Fx-
  - New: `values_out.append(raw)` for Fx-
- **Reason:** LINKED mode uses same Kt for both + and - directions
- **Updated:** Docstring to reflect new sign convention

#### `kt_optimizer/solver.py` (multiple locations)

**a) Lines 109-111: `_build_bounds()`**
- **Change:** Enforce Kt ≥ 0 constraint
  - Old: `return [(None, None)] * n_vars` (unbounded)
  - New: `return [(0, None)] * n_vars` (Kt ≥ 0)
- **Reason:** Kt is a geometric property, always non-negative

**b) Lines 86-92: Force matrix for SET mode**
- **Change:** Preserve sign in negative force column
  - Old: `f_neg = np.abs(np.minimum(f[:, i], 0.0))` (strips sign → ≥ 0)
  - New: `f_neg = np.minimum(f[:, i], 0.0)` (preserves sign → ≤ 0)
- **Reason:** Allow Kt ≥ 0 while preserving stress signs

**c) Lines 93-102: Force matrix for INDIVIDUAL mode**
- **Change:** Same as SET mode (preserve sign in f_neg)
  - Old: `f_neg = np.abs(np.minimum(f[:, i], 0.0))`
  - New: `f_neg = np.minimum(f[:, i], 0.0)`
- **Reason:** Consistency with SET mode

**d) Line 188: `_signed_kt_sigma()` validation for SET mode**
- **Change:** Use signed force
  - Old: `s += k_val * abs(fval)` (magnitude)
  - New: `s += k_val * fval` (signed)
- **Reason:** Consistency with force matrix

#### `tests/test_solver.py` (lines 111-209)

**Replaced test:**
- Old: `test_negative_kt_values_propagate_when_unconstrained()`
  - Tested sign-flip behavior with negative Kt values
  - No longer valid under new physics

**New tests:**
1. `test_linked_mode_same_kt_for_both_signs()`
   - Verifies: Kt_Fx+ = Kt_Fx- (same value, not negated)
   - Verifies: Both ≥ 0
   - Verifies: Stress signs preserved (tension +, compression -)

2. `test_individual_mode_asymmetric_kt()`
   - Verifies: Kt+ and Kt- can differ
   - Verifies: Both ≥ 0
   - Verifies: Stress signs preserved

3. `test_kt_nonnegativity_enforced()`
   - Verifies: Solver enforces Kt ≥ 0
   - Tests pathological case (opposite force/stress signs)
   - Expected: infeasible or violates conservative constraint

**Fixed test data:**
- `_sample_df()`: Changed LC4 from Fx=-100, σ=80 to Fx=-100, σ=-80
- `test_set_mode_fixed_values_contribute_to_prediction()`: Changed LC2 from Fx=-10, σ=15 to Fx=-10, σ=-15
- **Reason:** Old data had opposite signs for force and stress (physically inconsistent)

#### `CLAUDE.md` (lines 45-70)

**Added:** New "Sign Conventions" section documenting:
- Kt ≥ 0 always (physical property)
- Signed forces and stresses
- LINKED mode: σ = Kt × F (same Kt for both directions)
- INDIVIDUAL mode: σ = Kt+ × max(F,0) + Kt- × min(F,0) (separate Kt)
- Physical examples for each mode

## Physical Interpretation

### LINKED Mode (Symmetric)
```
Single Kt per component, same for tension and compression
σ = Kt × F

Example: Kt_Fx = 2.0
  F = +100 → σ = 2.0 × 100 = +200 (tension)
  F = -100 → σ = 2.0 × (-100) = -200 (compression)
```

### INDIVIDUAL Mode (Asymmetric)
```
Separate Kt+ and Kt- per component (both ≥ 0)
σ = Kt+ × max(F,0) + Kt- × min(F,0)

Example: Kt_Fx+ = 2.0, Kt_Fx- = 3.0
  F = +100 → σ = 2.0 × 100 + 3.0 × 0 = +200
  F = -100 → σ = 2.0 × 0 + 3.0 × (-100) = -300

Interpretation: Compression has higher stress concentration (Kt- = 3.0)
than tension (Kt+ = 2.0) for this geometry.
```

## Breaking Changes

⚠️ **This fix intentionally breaks backward compatibility:**

1. **Kt values will differ from previous versions**
   - Old: Could be negative, especially in LINKED mode
   - New: Always non-negative

2. **Display behavior changed**
   - Old: LINKED mode showed opposite signs (Fx+ = k, Fx- = -k)
   - New: LINKED mode shows same value (Fx+ = k, Fx- = k)

3. **Physically inconsistent data will fail**
   - Cases where force and stress have opposite signs
   - Previously: Solver found negative Kt
   - Now: Infeasible or violates conservative constraint

## Verification

### Test Results
```bash
uv run pytest tests/ -v
```
✅ All 14 tests pass
✅ Linter checks pass (ruff)

### Manual Verification Recommended

**Test 1: LINKED mode with consistent data**
```
LC1: Fx = +100, σ = +200 → expect Kt_Fx = 2.0
LC2: Fx = -100, σ = -200 → expect Kt_Fx = 2.0
Result: Kt_Fx+ = Kt_Fx- = 2.0 (same value, both positive)
```

**Test 2: INDIVIDUAL mode with asymmetric behavior**
```
LC1: Fx = +100, σ = +200 → expect Kt_Fx+ = 2.0
LC2: Fx = -100, σ = -300 → expect Kt_Fx- = 3.0
Result: Kt_Fx+ ≈ 2.0, Kt_Fx- ≈ 3.0 (different, both positive)
```

**Test 3: Invalid data (opposite signs) should fail**
```
LC1: Fx = -100, σ = +200 → physically inconsistent
Result: Infeasible or large underprediction
```

## Migration Notes

If you have existing CSV files with physically inconsistent data:
1. Review cases where force and stress have opposite signs
2. Check for data entry errors
3. Consider if the physics actually supports the observed behavior
4. For valid asymmetric cases, use INDIVIDUAL mode with appropriate sign modes

## Benefits

1. **Physical consistency:** All Kt values are non-negative (geometric property)
2. **Mathematical clarity:** σ = Kt × F throughout (no hidden sign flips)
3. **Predictable behavior:** Same physics in LP and validation
4. **Better diagnostics:** Invalid data is caught rather than hidden

## Files Changed Summary

- `kt_optimizer/models.py`: 1 function, docstring updates
- `kt_optimizer/solver.py`: 4 locations (bounds, force matrix, validation)
- `tests/test_solver.py`: 3 new tests, 2 test data fixes
- `CLAUDE.md`: Added sign conventions section
- All changes are tested and verified
