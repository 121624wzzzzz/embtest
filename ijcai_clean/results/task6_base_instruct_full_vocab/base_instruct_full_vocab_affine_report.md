# Base-Instruct Full-Vocabulary Affine Results

Source: `configs/base_instruct_pairs.yaml`, recomputed with every vocabulary row (`0..vocab_size-1`) for each Base-Instruct pair.

This run does not sample token rows. It fits the same centered affine relation `Y ~= X * A + b`, but computes it through streaming centered normal equations so full vocabularies can be handled without materializing one huge design matrix on GPU. It also records compact diagnostics for `A` instead of saving the full matrices.

## Summary

| group | n | R2_E mean | R2_U mean | rel_err_E mean | rel_err_U mean | E rel A-I mean | E identity cosine mean | E offdiag/A mean | E_delta rank95 mean | A-I rank95 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all Base-Instruct | 35 | 0.9434 | 0.9433 | 0.1348 | 0.1363 | 0.0835 | 0.9805 | 0.0735 | 2779.9143 | 1626.1429 |
| Gemma only | 11 | 0.8305 | 0.8305 | 0.3149 | 0.3149 | 0.2013 | 0.9394 | 0.1804 | 2827.2727 | 1324.2727 |
| Llama only | 4 | 0.9880 | 0.9874 | 0.0889 | 0.0942 | 0.0584 | 0.9982 | 0.0465 | 3709.5000 | 2580.0000 |
| Qwen only | 18 | 0.9963 | 0.9961 | 0.0497 | 0.0515 | 0.0263 | 0.9996 | 0.0221 | 2148.2222 | 1350.8333 |

## A Diagnostics

- `E_rel_A_minus_I_over_I`: `||A - I||_F / ||I||_F`; lower means closer to identity.
- `E_identity_cosine`: `trace(A) / (||A||_F ||I||_F)`; closer to 1 means A points in the identity direction.
- `E_offdiag_norm_over_A`: off-diagonal Frobenius norm divided by `||A||_F`; higher means more coordinate mixing.
- `E_rel_orthogonality_error_over_I`: `||A^T A - I||_F / ||I||_F`; lower means closer to an orthogonal rotation/reflection.
- `E_delta_rank95` and `A-I_rank95`: smallest rank explaining 95% of squared singular-value energy.
- `energy_at_1pct_h` / `energy_at_5pct_h` / `energy_at_10pct_h`: cumulative energy explained by the top 1% / 5% / 10% of hidden dimensions, for dimension-normalized spectrum comparison.

## Details

| model_a | model_b | tied A/B | vocab | hidden | R2_E | rel_err_E | R2_U | rel_err_U | E rel A-I/I | E I cosine | E offdiag/A | E orth err/I | E_delta rank95 | A-I rank95 | E_delta energy@5%h | A-I energy@5%h | elapsed sec |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DeepSeek-V3-Base` | `DeepSeek-V3` | False/False | 129280 | 7168 | 1.0000 | 0.0022 | 1.0000 | 0.0014 | 0.0008 | 1.0000 | 0.0008 | 0.0011 | 6317 | 3828 | 0.1930 | 0.2789 | 96.7 |
| `DeepSeek-V3.1-Base` | `DeepSeek-V3.1` | False/False | 129280 | 7168 | 1.0000 | 0.0038 | 1.0000 | 0.0032 | 0.0013 | 1.0000 | 0.0013 | 0.0019 | 6374 | 3885 | 0.1609 | 0.2559 | 78.9 |
| `Qwen2.5-32B` | `Qwen2.5-32B-Instruct` | False/False | 152064 | 5120 | 0.9999 | 0.0111 | 0.9997 | 0.0165 | 0.0072 | 1.0000 | 0.0071 | 0.0103 | 2450 | 1562 | 0.4332 | 0.5052 | 77.3 |
| `Qwen2.5-14B` | `Qwen2.5-14B-Instruct` | False/False | 152064 | 5120 | 0.9998 | 0.0127 | 0.9997 | 0.0169 | 0.0075 | 1.0000 | 0.0074 | 0.0107 | 2637 | 1567 | 0.3898 | 0.4830 | 67.1 |
| `Qwen2.5-3B` | `Qwen2.5-3B-Instruct` | True/True | 151936 | 2048 | 0.9997 | 0.0158 | 0.9997 | 0.0158 | 0.0179 | 0.9998 | 0.0172 | 0.0247 | 1290 | 120 | 0.7550 | 0.9469 | 11.5 |
| `Qwen2.5-7B` | `Qwen2.5-7B-Instruct` | False/False | 152064 | 3584 | 0.9997 | 0.0166 | 0.9992 | 0.0264 | 0.0067 | 1.0000 | 0.0066 | 0.0096 | 1761 | 1146 | 0.4073 | 0.4853 | 36.1 |
| `Qwen2.5-72B-Base` | `Qwen2.5-72B-Instruct` | False/False | 152064 | 8192 | 0.9997 | 0.0176 | 0.9993 | 0.0257 | 0.0088 | 1.0000 | 0.0087 | 0.0125 | 6405 | 3834 | 0.3410 | 0.3703 | 161.4 |
| `Qwen2.5-1.5B` | `Qwen2.5-1.5B-Instruct` | True/True | 151936 | 1536 | 0.9997 | 0.0169 | 0.9997 | 0.0169 | 0.0189 | 0.9998 | 0.0181 | 0.0263 | 887 | 98 | 0.7910 | 0.9448 | 6.3 |
| `Gemma-2-2B` | `Gemma-2-2B-Instruct` | True/True | 256000 | 2304 | 0.9993 | 0.0235 | 0.9993 | 0.0235 | 0.0543 | 0.9985 | 0.0534 | 0.0965 | 1561 | 29 | 0.6889 | 0.9778 | 22.2 |
| `Qwen3.5-35B-A3B-Base` | `Qwen3.5-35B-A3B-Instruct` | False/False | 248320 | 2048 | 0.9987 | 0.0355 | 0.9988 | 0.0325 | 0.0054 | 1.0000 | 0.0038 | 0.0094 | 1814 | 1364 | 0.1822 | 0.1831 | 25.6 |
| `Qwen3.5-9B-Base` | `Qwen3.5-9B-Instruct` | False/False | 248320 | 4096 | 0.9986 | 0.0373 | 0.9982 | 0.0403 | 0.0067 | 1.0000 | 0.0053 | 0.0111 | 3586 | 2558 | 0.1865 | 0.2059 | 98.0 |
| `Llama-3.1-70B-Base` | `Llama-3.1-70B-Instruct` | False/False | 128256 | 8192 | 0.9986 | 0.0378 | 0.9974 | 0.0505 | 0.0134 | 0.9999 | 0.0125 | 0.0201 | 7011 | 4577 | 0.1912 | 0.2508 | 89.1 |
| `Llama-3.1-8B` | `Llama-3.1-8B-Instruct` | False/False | 128256 | 4096 | 0.9984 | 0.0392 | 0.9976 | 0.0477 | 0.0094 | 1.0000 | 0.0080 | 0.0151 | 3647 | 2489 | 0.1433 | 0.2008 | 34.1 |
| `Qwen3.5-0.8B-Base` | `Qwen3.5-0.8B-Instruct` | True/True | 248320 | 1024 | 0.9979 | 0.0415 | 0.9979 | 0.0415 | 0.0177 | 0.9999 | 0.0146 | 0.0302 | 872 | 485 | 0.2723 | 0.6928 | 6.9 |
| `Gemma-2-27B` | `Gemma-2-27B-Instruct` | True/True | 256000 | 4608 | 0.9976 | 0.0484 | 0.9976 | 0.0484 | 0.0991 | 0.9951 | 0.0981 | 0.3798 | 3676 | 43 | 0.5699 | 0.9843 | 76.7 |
| `Qwen3.5-4B-Base` | `Qwen3.5-4B-Instruct` | True/True | 248320 | 2560 | 0.9972 | 0.0492 | 0.9972 | 0.0492 | 0.0201 | 0.9999 | 0.0168 | 0.0339 | 2172 | 1176 | 0.2739 | 0.6483 | 24.5 |
| `Qwen3.5-2B-Base` | `Qwen3.5-2B-Instruct` | True/True | 248320 | 2048 | 0.9966 | 0.0526 | 0.9966 | 0.0526 | 0.0220 | 0.9998 | 0.0183 | 0.0368 | 1738 | 949 | 0.2733 | 0.6698 | 16.0 |
| `Qwen3-8B-Base` | `Qwen3-8B` | False/False | 151936 | 4096 | 0.9957 | 0.0652 | 0.9942 | 0.0740 | 0.0251 | 0.9998 | 0.0213 | 0.0397 | 2971 | 2287 | 0.2841 | 0.3427 | 46.3 |
| `Qwen3-14B-Base` | `Qwen3-14B` | False/False | 151936 | 5120 | 0.9956 | 0.0660 | 0.9937 | 0.0785 | 0.0378 | 0.9994 | 0.0358 | 0.0556 | 2860 | 2031 | 0.4086 | 0.4539 | 63.8 |
| `Qwen3-1.7B-Base` | `Qwen3-1.7B` | True/True | 151936 | 2048 | 0.9938 | 0.0767 | 0.9938 | 0.0767 | 0.0585 | 0.9986 | 0.0509 | 0.0904 | 1691 | 1060 | 0.4068 | 0.7754 | 18.5 |
| `Qwen3-30B-A3B-Base` | `Qwen3-30B-A3B` | False/False | 151936 | 2048 | 0.9921 | 0.0884 | 0.9945 | 0.0716 | 0.0194 | 0.9999 | 0.0105 | 0.0358 | 1778 | 1555 | 0.1310 | 0.1560 | 19.2 |
| `Gemma-2-9B` | `Gemma-2-9B-Instruct` | True/True | 256000 | 3584 | 0.9911 | 0.0828 | 0.9911 | 0.0828 | 0.0511 | 0.9987 | 0.0507 | 0.0696 | 3123 | 941 | 0.4647 | 0.8435 | 31.6 |
| `Qwen2.5-0.5B` | `Qwen2.5-0.5B-Instruct` | True/True | 151936 | 896 | 0.9903 | 0.0908 | 0.9903 | 0.0908 | 0.0565 | 0.9989 | 0.0452 | 0.0872 | 798 | 583 | 0.3625 | 0.6658 | 4.6 |
| `Qwen3-4B-Base` | `Qwen3-4B` | True/True | 151936 | 2560 | 0.9901 | 0.0954 | 0.9901 | 0.0954 | 0.0645 | 0.9984 | 0.0544 | 0.0970 | 2109 | 1320 | 0.4042 | 0.7166 | 22.9 |
| `Qwen3-0.6B-Base` | `Qwen3-0.6B` | True/True | 151936 | 1024 | 0.9877 | 0.1050 | 0.9877 | 0.1050 | 0.0729 | 0.9982 | 0.0563 | 0.1166 | 849 | 620 | 0.4184 | 0.6688 | 12.8 |
| `Llama-3.2-3B` | `Llama-3.2-3B-Instruct` | True/True | 128256 | 3072 | 0.9795 | 0.1323 | 0.9795 | 0.1323 | 0.0992 | 0.9969 | 0.0772 | 0.1600 | 2506 | 1945 | 0.4748 | 0.5819 | 19.8 |
| `Llama-3.2-1B` | `Llama-3.2-1B-Instruct` | True/True | 128256 | 2048 | 0.9753 | 0.1461 | 0.9753 | 0.1461 | 0.1114 | 0.9960 | 0.0881 | 0.1771 | 1674 | 1309 | 0.4939 | 0.6074 | 12.7 |
| `Gemma-3-27B` | `Gemma-3-27B-Instruct` | True/True | 262208 | 5376 | 0.9708 | 0.1701 | 0.9708 | 0.1701 | 0.0450 | 0.9992 | 0.0387 | 0.0706 | 4815 | 3004 | 0.2007 | 0.4228 | 110.3 |
| `Gemma-3-12B` | `Gemma-3-12B-Instruct` | True/True | 262208 | 3840 | 0.9673 | 0.1794 | 0.9673 | 0.1794 | 0.0503 | 0.9992 | 0.0404 | 0.0815 | 3473 | 2272 | 0.2028 | 0.4488 | 42.2 |
| `Gemma-3-4B` | `Gemma-3-4B-Instruct` | True/True | 262208 | 2560 | 0.9579 | 0.2017 | 0.9579 | 0.2017 | 0.0529 | 0.9986 | 0.0509 | 0.0786 | 2329 | 1085 | 0.2129 | 0.7055 | 24.7 |
| `Gemma-4-31B` | `Gemma-4-31B-Instruct` | True/True | 262144 | 5376 | 0.7759 | 0.4641 | 0.7759 | 0.4641 | 0.2163 | 0.9763 | 0.2126 | 0.6466 | 4856 | 2060 | 0.2075 | 0.7260 | 119.2 |
| `Gemma-4-E4B` | `Gemma-4-E4B-Instruct` | True/True | 262144 | 2560 | 0.7247 | 0.4923 | 0.7247 | 0.4923 | 0.1568 | 0.9879 | 0.1529 | 0.2568 | 2342 | 1401 | 0.2024 | 0.3780 | 23.6 |
| `Gemma-4-26B-A4B` | `Gemma-4-26B-A4B-Instruct` | True/True | 262144 | 2816 | 0.7071 | 0.5169 | 0.7071 | 0.5169 | 0.3539 | 0.9828 | 0.1788 | 0.8592 | 2557 | 1968 | 0.2034 | 0.2852 | 21.3 |
| `Gemma-4-E2B` | `Gemma-4-E2B-Instruct` | True/True | 262144 | 1536 | 0.6682 | 0.5308 | 0.6682 | 0.5308 | 0.2237 | 0.9766 | 0.2110 | 0.4418 | 1371 | 833 | 0.3282 | 0.5276 | 13.9 |
| `Gemma-3-1B` | `Gemma-3-1B-Instruct` | True/True | 262144 | 1152 | 0.3751 | 0.7536 | 0.3751 | 0.7536 | 0.9110 | 0.4208 | 0.8973 | 0.7726 | 997 | 931 | 0.1942 | 0.1322 | 8.8 |
