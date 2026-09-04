# Morpheme Workspace Rules

## Dictionary Suffix and Plural Sourcing Rules
When adding definitions to the dictionary database, word lists, or dictionary text files (e.g., Added Words, `newNWL.txt`, and `newCSW` files):
1. **Noun Plurals**: If the definition of a word reads `"(Noun) plural of [singular word]"` (or similar), locate the singular word's definition in the dictionary and repeat it for the plural entry.
2. **Verb Conjugations**: Likewise, for conjugated verb endings (`-S`, `-ED`, `-ING`), locate the base root verb's definition and repeat it for the conjugated entry.
3. **Missing Definitions**: If the singular root word or base verb does not have a definition, research it and provide a clear, professional lexicographical definition.

## Mobile Fullscreen & Virtual Keyboard Invariant Rules (STRICT / PERMANENT)
1. **Full List Modal (`openFullListModal`)**: Whenever the full list modal is opened, the app MUST explicitly and immediately exit fullscreen (`document.exitFullscreen()`). Never remove or disable this logic in `static/js/tools.js`.
2. **Android Virtual Keyboard Black Screen Prevention**: On mobile (especially Android Chrome), triggering the soft keyboard while the browser is in fullscreen forces an OS display surface rebuild, causing a severe 2–3 second black screen. Therefore:
   - Fullscreen MUST be exited when navigating to non-game utility pages (Tools, Settings, Profile, Forum, How to Play, Donate) or opening modal dialogs with text inputs.
   - Automatic fullscreen re-engagement MUST NEVER trigger while on Tools, Settings, Profile, Forum, or when any modal/input is active.
3. **Gateway Screen (`#page-loading`)**: Fullscreen is requested immediately on any tap/press across the initial ENTER LOBBY screen so the Android notice appears on the gateway screen, and fullscreen is preserved continuously into the Lobby (`page-lobby`) without exiting or shifting layout dimensions.
