---
description: after every code change, stage all modified files, commit with a descriptive message, and push to origin main
---

After making any code change to the project, immediately run the following steps:

// turbo-all
1. Stage all changes:
```
git -C c:\stock_scanner add -A
```

// turbo-all
2. Commit with a short, descriptive message summarising the change (replace `<message>` with an appropriate one-line description):
```
git -C c:\stock_scanner commit -m "<message>"
```

// turbo-all
3. Push to origin main:
```
git -C c:\stock_scanner push origin main
```

If the commit returns "nothing to commit", skip steps 2 and 3.
