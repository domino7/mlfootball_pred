import csv
import datetime

if __name__ == "__main__":
	with open('Team_Attributes') as team_attributes_file, open('Match.csv') as match_file, open('learning_vectors3.csv', 'w+') as output:
		fieldnames = ['Match_id', 
						'League_id', 
						'Season', 
						'Stage', 
						'Date', 
						'home_team_api_id', 
						'away_team_api_id', 
						'Result', 
						'B365H', 'B365D' ,'B365A' ,
						'HomeTeam_buildUpPlaySpeed' ,
						'HomeTeam_buildUpPlayPassing', 
						'HomeTeam_chanceCreationShooting', 
						'HomeTeam_defencePressure', 
						'HomeTeam_buildUpPlayDribbling',
						'HomeTeam_chanceCreationPassing',
						'HomeTeam_chanceCreationCrossing',
						'HomeTeam_defenceAggression',
						'HomeTeam_defenceTeamWidth',
						'AwayTeam_buildUpPlaySpeed', 
						'AwayTeam_buildUpPlayPassing', 
						'AwayTeam_chanceCreationShooting', 
						'AwayTeam_defencePressure',
						'AwayTeam_buildUpPlayDribbling',
						'AwayTeam_chanceCreationPassing',
						'AwayTeam_chanceCreationCrossing',
						'AwayTeam_defenceAggression',
						'AwayTeam_defenceTeamWidth',]
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

			team_attributes_file.seek(0)
			team_attributes_reader = csv.DictReader(team_attributes_file)
			for row1 in team_attributes_reader:
				if row1['team_api_id'] == home_team_id:
					home_team_attributes.append(row1)
				if row1['team_api_id'] == away_team_id:
					away_team_attributes.append(row1)

			home_delta = 10000
			for row2 in home_team_attributes:
				tmp = abs(match_date - datetime.datetime.strptime(row2['date'], '%Y-%m-%d %H:%M:%S').date())
				if tmp < datetime.timedelta(days=home_delta):
					home_delta = tmp.days
					best_home_team_attribute_row = row2

			away_delta = 10000
			for row3 in away_team_attributes:
				tmp = abs(datetime.datetime.strptime(row3['date'], '%Y-%m-%d %H:%M:%S').date() - match_date)
				if tmp < datetime.timedelta(days=away_delta):
					away_delta = tmp.days
					best_away_team_attribute_row = row3

			
			writer.writerow({	'Match_id': match_row['id'], 
								'League_id' : match_row['league_id'],
								'Season': match_row['season'], 
								'Stage': match_row['stage'], 
								'Date': match_date, 
								'home_team_api_id': home_team_id, 
								'away_team_api_id': away_team_id,
								'Result': result, 
								'B365H': match_row['B365H'], 
								'B365D': match_row['B365D'] ,
								'B365A': match_row['B365A'],
								'HomeTeam_buildUpPlaySpeed' : best_home_team_attribute_row['buildUpPlaySpeed'], 
								'HomeTeam_buildUpPlayPassing' : best_home_team_attribute_row['buildUpPlayPassing'], 
								'HomeTeam_chanceCreationShooting' : best_home_team_attribute_row['chanceCreationShooting'], 
								'HomeTeam_defencePressure' : best_home_team_attribute_row['defencePressure'],
								'HomeTeam_buildUpPlayDribbling': best_home_team_attribute_row['buildUpPlayDribbling'],
								'HomeTeam_chanceCreationPassing': best_home_team_attribute_row['chanceCreationPassing'],
								'HomeTeam_chanceCreationCrossing': best_home_team_attribute_row['chanceCreationCrossing'],
								'HomeTeam_defenceAggression': best_home_team_attribute_row['defenceAggression'],
								'HomeTeam_defenceTeamWidth': best_home_team_attribute_row['defenceTeamWidth'],													
								'AwayTeam_buildUpPlaySpeed' : best_away_team_attribute_row['buildUpPlaySpeed'], 
								'AwayTeam_buildUpPlayPassing' : best_away_team_attribute_row['buildUpPlayPassing'], 
								'AwayTeam_chanceCreationShooting' : best_away_team_attribute_row['chanceCreationShooting'], 
								'AwayTeam_defencePressure' : best_away_team_attribute_row['defencePressure'],
								'AwayTeam_buildUpPlayDribbling': best_away_team_attribute_row['buildUpPlayDribbling'],
								'AwayTeam_chanceCreationPassing': best_away_team_attribute_row['chanceCreationPassing'],
								'AwayTeam_chanceCreationCrossing': best_away_team_attribute_row['chanceCreationCrossing'],
								'AwayTeam_defenceAggression': best_away_team_attribute_row['defenceAggression'],
								'AwayTeam_defenceTeamWidth': best_away_team_attribute_row['defenceTeamWidth'] })

		# team_attributes_reader = csv.DictReader(team_attributes_file)
		# for row in team_attributes_reader:
		# 	print(row['team_fifa_api_id'], row['team_api_id'])

		