from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.results = []

    def handle_starttag(self, tag, attrs):
        if tag in ['div', 'main']:
            attrs_dict = dict(attrs)
            id_val = attrs_dict.get('id', tag)
            self.results.append((id_val, list(self.stack)))
            self.stack.append(id_val)

    def handle_endtag(self, tag):
        if tag in ['div', 'main']:
            if self.stack:
                self.stack.pop()

parser = MyHTMLParser()
with open('templates/index.html', 'r') as f:
    parser.feed(f.read())

for item, stack in parser.results:
    if item.startswith('page-'):
        parent = stack[-1] if stack else "None"
        print(f"{item} is child of {parent}")
