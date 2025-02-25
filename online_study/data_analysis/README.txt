Folder `desensitized data` contains:
- Desensitized_data_from_597_participants.csv: Desensitized raw data. 
- Q7_annotated_three_scenarios_llms.csv: the combined results from the two LLM annotators 
- Q7_annotated_three_scenarios_llms+human.csv: a human annotator checked the results from LLM annotators and decide the final annotations
- Scenario_A/B/C_Q5/Q6/Q7.csv: extracted open-ended answers of Q5,Q6,Q7 (use the original question index from the questionnaire)
- Scenario_A/B/C_Q7_o3/Gemini.csv: the annnotated opinions of Q7 from the two LLM annotators   

Prompt and answers from ChatGPT o3:
- prompt.txt

Notebooks for data process and analysis:
- data_process.ipynb: generate the csv files in `desensitized data` the from the raw data
- data_analysis.ipynb: data analysis and visualization