Phase: Level 1, a tree of training configs.

1. Profile: call profile_dataset, look at the images and stats, then save_profile_notes with what matters for modelling (bands, class balance, what looks hard).
2. Plan: call list_backbones and get_budget. If the playbook tools are available, call read_playbook, then compose_config. If compose_config returns a first config, launch it unchanged as your first decision run. If it also returns a re-test partner, launch that as your second decision run. Otherwise, draft the first config yourself and pass it to compose_config(draft=...): code may return a re-test partner for it, which you launch as your second decision run.
3. Each launch states a hypothesis: what you expect and why. Give parent_run_id for an improve run.
4. While a run trains, call wait_run. Evaluate finished runs with evaluate_run, look at the worst mistakes, then record_diagnosis: what the errors suggest you should try next.
5. The harness repeats your first finished decision run with another seed to measure noise. A change is kept only if its gain beats the measured noise. Read each run's outcome: kept, within_noise, regressed or failed.
6. Screens (at most two) only rank options. They never become the incumbent.
7. After three runs in a row with no kept gain, diversify: siblings of the incumbent, or new seeds of good recipes.
8. Continue until get_budget shows the Level 1 pot cannot fit another decision run. Then stop and reply with a short summary. This phase has no finalize.
