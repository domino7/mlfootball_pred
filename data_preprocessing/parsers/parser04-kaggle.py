import csv
import datetime

if __name__ == "__main__":
	with open('learning_data01.csv') as learning_vector, open('teamsAgeTMV.csv') as teams_age, open('learning_vectors7.csv', 'w+') as output:
		fieldnames = ['Match_id','League_id','Season','Stage','Date','H_team','A_team','Result','B365H','B365D','B365A','H_Speed','H_Pass','H_Shoot','H_Pressure','H_chPass','H_chCross','H_dAggr','H_dWidth','A_Speed','A_Pass','A_Shoot','A_Pressure','A_chPass','A_chCross','A_dAggr','A_dWidth', 'H_age', 'A_age', 'H_TMV', 'A_TMV']
		writer = csv.DictWriter(output, lineterminator='\n', fieldnames=fieldnames)
		writer.writeheader()
		match_reader = csv.DictReader(learning_vector)
		age_reader = csv.DictReader(teams_age)

		for match_row in match_reader:
			#print match_row
			h_age = 0
			a_age = 0
			h_tmv = 0
			a_tmv = 0
			year = datetime.datetime.strptime(match_row['Date'], '%Y-%m-%d').year
			month = datetime.datetime.strptime(match_row['Date'], '%Y-%m-%d').month
			if (month < 7):
				year = year - 1
			year = str(year)
			year = year[-2:]
			field_name = "Age" + year
			tmv_field_name = "TMV" + year
			#print field_name

			teams_age.seek(0)
			for row in age_reader:
				if match_row['H_team'].lower() == row['Club'].lower():
					#print row[field_name]
					h_age = row[field_name]
					h_tmv = row[tmv_field_name]
				if match_row['A_team'].lower() == row['Club'].lower():
					#print row[field_name]
					a_age = row[field_name]
					a_tmv = row[tmv_field_name]

			match_row['H_age'] = h_age
			match_row['A_age'] = a_age
			match_row['H_TMV'] = h_tmv
			match_row['A_TMV'] = a_tmv
			
			if h_age == 0 or h_tmv == 0:
				print (match_row['H_team'])
			if a_age == 0 or a_tmv == 0:
				print (match_row['A_team'])

			writer.writerow(match_row)

			# home_team_id = match_row['home_team_api_id']
			# away_team_id = match_row['away_team_api_id']
			# match_date = datetime.datetime.strptime(match_row['date'], '%Y-%m-%d %H:%M:%S').date() 
			# home_goals = match_row['home_team_goal']
			# away_goals = match_row['away_team_goal']

			# if home_goals > away_goals:
			# 	result = 0
			# elif away_goals == home_goals:
			# 	result = 1
			# else:
			# 	result =2

			# home_team_attributes = []
			# away_team_attributes = []

			# team_attributes_file.seek(0)
			# team_attributes_reader = csv.DictReader(team_attributes_file)
			# for row1 in team_attributes_reader:
			# 	if row1['team_api_id'] == home_team_id:
			# 		home_team_attributes.append(row1)
			# 	if row1['team_api_id'] == away_team_id:
			# 		away_team_attributes.append(row1)

			# home_delta = 10000
			# for row2 in home_team_attributes:
			# 	tmp = abs(match_date - datetime.datetime.strptime(row2['date'], '%Y-%m-%d %H:%M:%S').date())
			# 	if tmp < datetime.timedelta(days=home_delta):
			# 		home_delta = tmp.days
			# 		best_home_team_attribute_row = row2

			# away_delta = 10000
			# for row3 in away_team_attributes:
			# 	tmp = abs(datetime.datetime.strptime(row3['date'], '%Y-%m-%d %H:%M:%S').date() - match_date)
			# 	if tmp < datetime.timedelta(days=away_delta):
			# 		away_delta = tmp.days
			# 		best_away_team_attribute_row = row3

			
			
		# team_attributes_reader = csv.DictReader(team_attributes_file)
		# for row in team_attributes_reader:
		# 	print(row['team_fifa_api_id'], row['team_api_id'])

		