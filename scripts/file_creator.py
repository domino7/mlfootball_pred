import csv
import pandas
import sys
import errno
import operator

# "CONSTANTS" used throughout the program for convienience
LEAGUE_ID_H = 'League_id'
SEASON_H = 'Season'
RESULT_H = 'Result'
H_NAME_H = 'H_team' 
A_NAME_H = 'A_team'
MATCH_ID_H = 'Match_id'
ENG_LEAGUE_ID = '1729'
SPAN_LEAGUE_ID = '21518'

# Improved, now gets the reverse sort
# If not enough matches are available -returns empty list
# TODO: Change ID to data sort
def getLastNMatchesOfATeam(data, teamName, N, match_id):
    team_matches = []
    for row in data:
        if ((row[H_NAME_H] == teamName) or (row[A_NAME_H] == teamName)) and (int(row[MATCH_ID_H]) < int(match_id)):
            team_matches.append(row)
    if(len(team_matches) < int(N)):
        print "not enough data"
        return []
    return sorted(team_matches, key=operator.itemgetter(MATCH_ID_H), reverse=True)[:N]        

# Calculate form of the team in a given set of matches
# ARGUMENTS:
# - matches     : a limited set of matches 
# - team_name   : name of the team taken into calculation
def calculateForm(matches, team_name):
    form = 0
    for row in matches:
        result = RESULTS[row[RESULT_H]]
        if (row[H_NAME_H] == team_name):
            if result == 'h_win':
                form += 3
            elif result == 'draw':
                form += 1
            elif result == 'a_win':
                form += 0
        elif(row[A_NAME_H] == team_name):
            if result == 'h_win':
                form += 0
            elif result == 'draw':
                form += 1
            elif result == 'a_win':
                form += 3
    return form

# Calculate the (weighted) mean of values
# If receives empty list - returns NA
def calculateMean(matches, team_name, attr_name_home, attr_name_away):
    if len(matches) == 0:
        return "NA"
    total = 0
    for row in matches:
        if row[H_NAME_H] == team_name:
            total += float(row[attr_name_home])
        elif row[A_NAME_H] == team_name:
            total += float(row[attr_name_away])
    return (total / len(matches))          

# Calculate the (weighted) sum of values
# If receives empty list - returns NA
def calculateSum(matches, team_name, attr_name_home, attr_name_away):
    if len(matches) == 0:
        return "NA"
    total = 0
    for row in matches:
        if row[H_NAME_H] == team_name:
            total += float(row[attr_name_home])
        elif row[A_NAME_H] == team_name:
            total += float(row[attr_name_away])
    return total

# Load CSV file with given name
def loadCSVFile(filename):
    data = []
    headers = []
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
        headers = reader.fieldnames
        for row in reader:
            data.append(row)
    return data, headers

# Write to CSV file with given names
def writeCSVFile(filename, headers, data):
    with open(filename, 'wb') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames = headers,  delimiter=',', quotechar='|')
        writer.writeheader()
        for row in data:
            writer.writerow(row)



if __name__ == "__main__":
    csvFile = sys.argv[1]
    data, headers = loadCSVFile(csvFile)
    
    for i in range(0, len(headers)):
        sys.stdout.write(( "["+ str(i) + "] " + headers[i]).ljust(35))
        if (i + 1) % 5 == 0:
            sys.stdout.flush()
            print ""
    print "\n"
    print "Please input desired headers as integers separted by spaces:"
    sys.stdout.write(">> ")
    numbers = map(int, raw_input().split())
    for number in numbers:
        if not (0 <= number < len(headers)):
            print "Error: Sorry, the number you have selected is outside list range!"
            sys.exit(errno.EINVAL)
    new_data = []
    new_headers = []
    print numbers
    for number in numbers:
        new_headers.append(headers[number])
    
    for row in data:
        new_row = {}
        for header in new_headers:
            new_row[header] = row[header]
        new_data.append(new_row)
        
    print new_headers
    while(True): 
        print "Would you like to calculate additional parameters (form, mean, etc.)? [Y/n]"
        sys.stdout.write(">> ")
        if raw_input().lower() != "y":
            print "Output file name: "
            outCsvFile = raw_input(">> ")
            writeCSVFile(outCsvFile, new_headers, new_data)
            print "Thanks, bye!"
            sys.exit(0)
        
        print "What should the new parameter be?"
        print "\t[1] A sum"
        print "\t[2] A mean"
        print "\t[3] A ratio"
        print "\t[4] A difference"
        print "\t[5] A form formula (provide values for win/draw/loose)"
        choice = int(raw_input(">> "))
        
        if choice == 1:
            print "Provide a name for the new parameter(HOME):"
            new_name_home = raw_input(">> ")
            print "Provide a name for the new parameter(AWAY):"
            new_name_away = raw_input(">> ")
            print "What parameter should I aggregate(HOME)?"
            h_name_home = raw_input(">> ")
            print "What parameter should I aggregate(AWAY)?"
            h_name_away = raw_input(">> ")
            print "How many matches whould I take into consideration?"
            no_matches = int(raw_input(">> "))
            for row in new_data:
                home_team_matches = getLastNMatchesOfATeam(data, row[H_NAME_H], no_matches, row[MATCH_ID_H])
                away_team_matches = getLastNMatchesOfATeam(data, row[A_NAME_H], no_matches, row[MATCH_ID_H])
                new_home_value = calculateSum(home_team_matches, row[H_NAME_H], h_name_home, h_name_away)
                new_away_value = calculateSum(away_team_matches, row[A_NAME_H], h_name_home, h_name_away)
                row[new_name_home] = new_home_value
                row[new_name_away] = new_away_value
            new_headers.append(new_name_home)
            new_headers.append(new_name_away)
        elif choice == 2:
            print "Provide a name for the new parameter(HOME):"
            new_name_home = raw_input(">> ")
            print "Provide a name for the new parameter(AWAY):"
            new_name_away = raw_input(">> ")
            print "What parameter should I aggregate(HOME)?"
            h_name_home = raw_input(">> ")
            print "What parameter should I aggregate(AWAY)?"
            h_name_away = raw_input(">> ")
            print "How many matches whould I take into consideration?"
            no_matches = int(raw_input(">> "))
            for row in new_data:
                home_team_matches = getLastNMatchesOfATeam(data, row[H_NAME_H], no_matches, row[MATCH_ID_H])
                away_team_matches = getLastNMatchesOfATeam(data, row[A_NAME_H], no_matches, row[MATCH_ID_H])
                new_home_value = calculateMean(home_team_matches, row[H_NAME_H], h_name_home, h_name_away)
                new_away_value = calculateMean(away_team_matches, row[A_NAME_H], h_name_home, h_name_away)
                row[new_name_home] = new_home_value
                row[new_name_away] = new_away_value
            new_headers.append(new_name_home)
            new_headers.append(new_name_away)
        elif choice == 3:
            print "Provide a name for the new parameter:"
            new_param_home = raw_input(">> ")
            print "What parameter should be in the nominator?"
            nom_header_name = raw_input(">> ")
            print "What parameter should be in the denominator?"
            denom_header_nme = raw_input(">> ")
            new_headers.append(new_param_home)
            for idx in range(0, len(data)):
                new_value = float(data[idx][nom_header_name]) / float(data[idx][denom_header_nme])
                new_data[idx][new_param_home] = new_value
        elif choice == 4:
            print "Provide a name for the new parameter:"
            new_param_home = raw_input(">> ")
            print "What parameter should be substracted from?"
            first_header_name = raw_input(">> ")
            print "What parameter should be substracted?"
            second_header_nme = raw_input(">> ")
            new_headers.append(new_param_home)
            for idx in range(0, len(data)):
                new_value = float(data[idx][first_header_name]) - float(data[idx][second_header_nme])
                new_data[idx][new_param_home] = new_value
        elif choice == 5:
            print "TBA"
        else:
            print "Sorry, I don't recognize that option"

    
