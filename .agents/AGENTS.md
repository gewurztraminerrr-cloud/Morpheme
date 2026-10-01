# Morpheme Workspace Rules

## Dictionary Suffix and Plural Sourcing Rules
When adding definitions to the dictionary database, word lists, or dictionary text files (e.g., Added Words, `newNWL.txt`, and `newCSW` files):
1. **Noun Plurals**: If the definition of a word reads `"(Noun) plural of [singular word]"` (or similar), locate the singular word's definition in the dictionary and repeat it for the plural entry.
2. **Verb Conjugations**: Likewise, for conjugated verb endings (`-S`, `-ED`, `-ING`), locate the base root verb's definition and repeat it for the conjugated entry.
3. **Missing Definitions**: If the singular root word or base verb does not have a definition, research it and provide a clear, professional lexicographical definition.

## Mobile Fullscreen & Virtual Keyboard Invariant Rules (STRICT / PERMANENT)
1. **Full List Modal (`openFullListModal`)**: Whenever the full list modal is opened, the app MUST explicitly and immediately exit fullscreen (`document.exitFullscreen()`). Never remove or disable this logic in `static/js/tools.js`.
2. **Mobile Continuous Fullscreen & Fast Navigation**: Fullscreen is preserved continuously across all pages, top menu tabs (Lobby, Play, Tools, Mods, Leaderboard, Settings, Profile, Forum, How to Play, Donate), Lobby Chat Drawer, textboxes, and dropdown menus. Fullscreen MUST NEVER be exited on textbox focus/blur, dropdown interactions, tab switching, or room entry/exit. This ensures fast, responsive navigation without OS "To exit full screen..." notices or bottom navigation bars shifting layout dimensions. Fullscreen exit is strictly isolated to the Full List Modal (`openFullListModal`).
3. **Gateway Screen (`#page-loading`)**: Fullscreen is requested synchronously upon pressing the gateway button (ENTER LOBBY / LOGIN), preserving continuous fullscreen into the Lobby and across the entire app without twitching or jumping the button on background touches.

