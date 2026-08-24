# Tier 0 verdict: does the acceptance probe read a distinct verification feature?

## E1 -- cheap-feature decomposition

**T=0.0**: peak M_h(MLP) AUROC = 0.9015 at layer 26. M_ABC (cheap-only) AUROC at that layer = 0.9530. ΔAUROC(M_offset - M_ABC) = 0.0052 (95% CI [0.0041, 0.0063]) → **NO_DISTINCT_FEATURE**.

**T=0.7**: peak M_h(MLP) AUROC = 0.8711 at layer 26. M_ABC (cheap-only) AUROC at that layer = 0.9352. ΔAUROC(M_offset - M_ABC) = 0.0089 (95% CI [0.0075, 0.0109]) → **NO_DISTINCT_FEATURE**.


## E2 -- two-position decomposition

T=0.0 P_dec: peak AUROC = 0.9015 at layer 26
T=0.0 P_tok: peak AUROC = 0.9586 at layer 27
T=0.7 P_dec: peak AUROC = 0.8711 at layer 26
T=0.7 P_tok: peak AUROC = 0.9491 at layer 27

Existing Phase 3 pipeline reads **P_dec** (confirmed by code inspection, Step 0) -- see NOTES.md for the position-math trace.


## E3 -- drafter-swap transfer

| domain     | position | model                          |  mean_transfer_ratio |
|------------|----------|--------------------------------|----------------------|
| chat       | P_dec    | transfer_ratio_A_to_B_linear   |               1.2094 |
| chat       | P_dec    | transfer_ratio_A_to_B_mlp      |               1.1126 |
| chat       | P_tok    | transfer_ratio_A_to_B_linear   |               1.2359 |
| chat       | P_tok    | transfer_ratio_A_to_B_mlp      |               1.1843 |
| code       | P_dec    | transfer_ratio_A_to_B_linear   |               1.0202 |
| code       | P_dec    | transfer_ratio_A_to_B_mlp      |               1.0200 |
| code       | P_tok    | transfer_ratio_A_to_B_linear   |               1.0141 |
| code       | P_tok    | transfer_ratio_A_to_B_mlp      |               1.0197 |
| reasoning  | P_dec    | transfer_ratio_A_to_B_linear   |               1.0221 |
| reasoning  | P_dec    | transfer_ratio_A_to_B_mlp      |               1.0254 |
| reasoning  | P_tok    | transfer_ratio_A_to_B_linear   |               1.0298 |
| reasoning  | P_tok    | transfer_ratio_A_to_B_mlp      |               1.0400 |