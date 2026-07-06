import app, json
print("Pre-warming dictionary...")
app.load_tools_dictionary("CSW")
ctx = app.app.test_request_context(path="/api/tools/combo", method="POST", json={"search_term":"GLOTTAL","dictionary":"CSW"})
ctx.push()
res = app.tools_combo_check()
data = res.get_json()
print("EPIGLOTTAL in 3MP?", "EPIGLOTTAL" in data["mp_groups"]["3"])
print("ISOGLOTTAL in 3MP?", "ISOGLOTTAL" in data["mp_groups"]["3"])
