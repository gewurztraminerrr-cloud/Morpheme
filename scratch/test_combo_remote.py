import app, json
print("Pre-warming dictionary...")
dict_data = app.load_tools_dictionary("CSW")
word_list = dict_data["words"]
print("EPIGLOTTAL in dictionary?", "EPIGLOTTAL" in dict_data["set"])
print("ISOGLOTTAL in dictionary?", "ISOGLOTTAL" in dict_data["set"])

ctx = app.app.test_request_context(path="/api/tools/combo", method="POST", json={"search_term":"GLOTTAL","dictionary":"CSW"})
ctx.push()
res = app.tools_combo_check()
data = res.get_json()
print("3MP group size:", len(data["mp_groups"]["3"]))
print("First 30 in 3MP:", data["mp_groups"]["3"][:30])
