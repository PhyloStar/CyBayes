import tree_helper
t="(a:1, (b:1, c:1):1);"
t = "((a:1, b:2):5, (c:3, d:4):6)"
t = "(((a:1,b:2):0.5,c:3):5,d:4)"
t = "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5)"
ed = tree_helper.newick2bl(t)
print(ed)
