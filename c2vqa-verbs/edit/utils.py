import json

def read_json(filename):
    print "Reading [%s]..." % (filename)
    with open(filename) as inputFile:
        jsonData = json.load(inputFile)
    print "Finished reading [%s]." % (filename)
    return jsonData