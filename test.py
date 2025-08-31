import twint

c = twint.Config()
c.Search = "#MANIT"
c.Limit = 100
c.Store_csv = True
c.Output = "tweets.csv"
twint.run.Search(c)
