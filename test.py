import pandas as pd 


data = pd.read_csv("D:\Project\prakriti_backup\cloud\csv\\nback_task\\nback.csv")  


second_last_row = data.iloc[-2]  # Second last row
last_row = data.iloc[-1]

true_1back = second_last_row[2:].to_list()
true_2back = last_row[2:].to_list()


def calculate_correct_percentage(row, true_answers):
    correct = (row[2:] == true_answers).sum()  # Compare answers excluding 'pid'
    total = len(true_answers)  # Total number of questions
    return (correct / total) * 100 

percentage_per_pid = []


for _, row in data.iterrows():
    pid = row.iloc[0]
    if pid != "True_1back" and pid != "True_2back":
        if row.iloc[1] == "1back":
            ans = calculate_correct_percentage(row, true_1back)
            
            percentage_per_pid.append((pid, "1back", ans)) 

        if row.iloc[1] == "2back":
            ans = calculate_correct_percentage(row, true_2back)

            percentage_per_pid.append((pid, "2back", ans)) 

percentage_df = pd.DataFrame(percentage_per_pid, columns =['PID', 'Task', 'Percentage']).to_csv("D:\Project\prakriti_backup\cloud\csv\\nback_task\\percentage_correct_nback.csv")