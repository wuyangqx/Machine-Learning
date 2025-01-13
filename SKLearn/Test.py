class TrieNode:
    def __init__(self):
        self.children = {}
        self.val = -1 

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def find(self, path): # find if a word is in the Trie
        cur = self.root
        for p in path:
            if p not in cur.children:
                return False
            cur = cur.children[p]
        return self.val

    def insert(self, path, val):
        cur = self.root
        for p in path:
            if p not in cur.children:
                cur.children[p] = TrieNode()
            cur = cur.children[p]
        cur.val = val

class FileSystem:
    def __init__(self):
        self.trie = Trie()
    
    def createPath(self, path, val):
        if self.trie.find(path) > 0:
            return False
        
        parent = path[:-1]
        if not self.trie.find(parent):
            return False
        
        self.trie.insert(path, val)
        return True
    
    def get(self, path):
        return self.trie.find(path)
