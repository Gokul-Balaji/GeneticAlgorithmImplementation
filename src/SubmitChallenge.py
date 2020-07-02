import requests
url='https://lf8q0kx152.execute-api.us-east-2.amazonaws.com/default/computeFitnessScore'
x=requests.post(url,json={"qconfig":"6 2 7 1 4 0 5 3",
                          "userID":736072,
                          "githubLink":"https://github.com/Gokul-Balaji/GeneticAlgorithmImplementation/tree/master/src"})
print(x.text)
