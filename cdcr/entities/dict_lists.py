from nltk.corpus import stopwords


class LocalDictLists:
    stopwords = set((stopwords.words('english'))).union({"'s", "\\'s", "the", "a", "of", "from", "``", "\'\'", "--"})
    pronouns = {'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his',
                     'himself', 'she','her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours',
                     'ourselves', 'you', 'your',  'yours', 'yourselves', 'they', 'them', 'their', 'theirs',
                     'themselves'}
    titles = {"Mr.", "Ms.", "Mrs.", "Dr.", "Prof."}
    people = {"people", "crowd", "public"}
    general_nouns = {"something", "anything", "nothing", "everything", "someone", "anyone", "everyone"}
    general_adj = {"most", "many", "other", "so-called", "more", "same", "former"}

