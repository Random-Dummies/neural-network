from __future__ import print_function
import json
def getScores(payload):
    scoreDict = {"skills":0,"experience":0,"CompanyReview":0,"CEOReview":0,"NeoApi":0}
    # JSON requires double-quotes, not single-quotes.
    payload = json.loads(payload)
    print(payload)
    for doc in payload['records']:
        scoreDict[doc['type']] = doc['score']
        print(doc['score'], doc['type'])
    scoreList = []
    scoreList.append(scoreDict["skills"])
    scoreList.append(scoreDict["experience"])
    scoreList.append(scoreDict["CompanyReview"])
    scoreList.append(scoreDict["CEOReview"])
    scoreList.append(scoreDict["NeoApi"])
    return scoreList
