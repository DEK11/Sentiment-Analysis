class Authentication:
    def __init__(self):
        cred = open("resources/MyCredentials.txt", "r", encoding='utf-8', errors='replace').read().split('\n')
        self.ckey = cred[0]
        self.csecret = cred[1]
        self.atoken = cred[2]
        self.asecret = cred[3]
