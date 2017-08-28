import csv
import datetime

if __name__ == "__main__":
	with open('Team_Attributes.csv') as team_attributes_file, open('Match.csv') as match_file, open('learning_vectors.csv', 'w+') as output:
		fieldnames = ['League_id', 'Season', 'Stage', 'Date' ,'Result', 'B365H', 'B365D' ,'B365A' ,'HomeTeam_buildUpPlaySpeed' ,'HomeTeam_buildUpPlayPassing', 'HomeTeam_chanceCreationShooting', 'HomeTeam_defencePressure', 'AwayTeam_buildUpPlaySpeed', 'AwayTeam_buildUpPlayPassing', 'AwayTeam_chanceCreationShooting', 'AwayTeam_defencePressure']
		writer = csv.DictWriter(output, fieldnames=fieldnames)
		writer.writeheader()
		match_reader = csv.DictReader(match_file)
		for match_row in match_reader:
			home_team_id = match_row['home_team_api_id']
			away_team_id = match_row['away_team_api_id']
			match_date = datetime.datetime.strptime(match_row['date'], '%Y-%m-%d %H:%M:%S').date() 
			home_goals = match_row['home_team_goal']
			away_goals = match_row['away_team_goal']

			if home_goals > away_goals:
				result = 0
			elif away_goals == home_goals:
				result = 1
			else:
				result =2

			home_team_attributes = []
			away_team_attributes = []

			team_attributes_reader = csv.DictReader(team_attributes_file)
			for row in team_attributes_reader:
				if row['team_api_id'] == home_team_id:
					home_team_attributes.append(row)
				if row['team_api_id'] == away_team_id:
					away_team_attributes.append(row)

			home_delta = 10000
			for row in home_team_attributes:
				tmp = abs(datetime.datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S').date() - match_date)
				print tmp
				if tmp < datetime.timedelta(days=home_delta):
					home_delta = tmp.days
					best_home_team_attribute_row = row
					print row

			away_delta = 10000
			for row in away_team_attributes:
				tmp = abs(datetime.datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S').date() - match_date)
				if tmp < datetime.timedelta(days=away_delta):
					away_delta = tmp.days
					best_away_team_attribute_row = row

			
			writer.writerow({	'League_id' : match_row['league_id'],'Season': match_row['season'], 'Stage': match_row['stage'], 'Date': match_date, 'Result': result, 'B365H': match_row['B365H'], 'B365D': match_row['B365D'] ,'B365A': match_row['B365A'],
												'HomeTeam_buildUpPlaySpeed' : best_home_team_attribute_row['buildUpPlaySpeed'], 'HomeTeam_buildUpPlayPassing' : best_home_team_attribute_row['buildUpPlayPassing'], 'HomeTeam_chanceCreationShooting' : best_home_team_attribute_row['chanceCreationShooting'], 'HomeTeam_defencePressure' : best_home_team_attribute_row['defencePressure'],
												'AwayTeam_buildUpPlaySpeed' : best_away_team_attribute_row['buildUpPlaySpeed'], 'AwayTeam_buildUpPlayPassing' : best_away_team_attribute_row['buildUpPlayPassing'], 'AwayTeam_chanceCreationShooting' : best_away_team_attribute_row['chanceCreationShooting'], 'AwayTeam_defencePressure' : best_away_team_attribute_row['defencePressure']})

		# team_attributes_reader = csv.DictReader(team_attributes_file)
		# for row in team_attributes_reader:
		# 	print(row['team_fifa_api_id'], row['team_api_id'])

		