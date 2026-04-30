// dxlang syntax highlighter — produces HTML spans from source text.

type Token = { text: string; className?: string };

export function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < source.length) {
    // Comment: # ... until end of line
    if (source[i] === '#') {
      const start = i;
      while (i < source.length && source[i] !== '\n') i++;
      tokens.push({ text: source.slice(start, i), className: 'tk-comment' });
      continue;
    }

    // String: "..."
    if (source[i] === '"') {
      const start = i;
      i++;
      while (i < source.length && source[i] !== '"') i++;
      i++;
      tokens.push({ text: source.slice(start, i), className: 'tk-string' });
      continue;
    }

    // Number (allow leading minus for negative numbers)
    if (/[0-9]/.test(source[i]) ||
        (source[i] === '-' && i + 1 < source.length && /[0-9.]/.test(source[i + 1]))) {
      const start = i;
      if (source[i] === '-') i++;
      while (i < source.length && /[0-9.eE+\-]/.test(source[i])) i++;
      tokens.push({ text: source.slice(start, i), className: 'tk-number' });
      continue;
    }

    // Type annotation: :identifier<...>
    if (source[i] === ':' && i + 1 < source.length && /[a-zA-Z_<>]/.test(source[i + 1])) {
      const start = i;
      i++;
      while (i < source.length && /[a-zA-Z0-9_<>]/.test(source[i])) i++;
      tokens.push({ text: source.slice(start, i), className: 'tk-type' });
      continue;
    }

    // Arrow: ->
    if (source[i] === '-' && i + 1 < source.length && source[i + 1] === '>') {
      tokens.push({ text: '->', className: 'tk-arrow' });
      i += 2;
      continue;
    }

    // Identifiers and keywords
    if (/[a-zA-Z_]/.test(source[i])) {
      const start = i;
      while (i < source.length && /[a-zA-Z0-9_]/.test(source[i])) i++;
      const word = source.slice(start, i);

      if (word === 'def') {
        tokens.push({ text: word, className: 'tk-keyword' });
      } else if (word === 'true' || word === 'false') {
        tokens.push({ text: word, className: 'tk-boolean' });
      } else if (
        word === 'int' || word === 'float' || word === 'bool' || word === 'string' ||
        word === 'tensor' || word === 'f32' || word === 'f16' || word === 'f64' ||
        word === 'i8' || word === 'i16' || word === 'i32' || word === 'i64'
      ) {
        tokens.push({ text: word, className: 'tk-type' });
      } else {
        tokens.push({ text: word, className: 'tk-identifier' });
      }
      continue;
    }

    // ./ path prefix
    if (source[i] === '.' && i + 1 < source.length && source[i + 1] === '/') {
      tokens.push({ text: './', className: 'tk-path' });
      i += 2;
      continue;
    }

    // Single dot
    if (source[i] === '.') {
      tokens.push({ text: '.', className: 'tk-dot' });
      i++;
      continue;
    }

    // Braces/parens
    if ('{}()[]'.includes(source[i])) {
      tokens.push({ text: source[i], className: 'tk-bracket' });
      i++;
      continue;
    }

    // Operators
    if ('+-*/%=<>&|!^'.includes(source[i])) {
      tokens.push({ text: source[i], className: 'tk-operator' });
      i++;
      continue;
    }

    // Whitespace and everything else
    tokens.push({ text: source[i] });
    i++;
  }

  return tokens;
}

export function highlightToHtml(source: string): string {
  return tokenize(source)
    .map((t) => {
      const escaped = t.text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      if (t.className) {
        return `<span class="${t.className}">${escaped}</span>`;
      }
      return escaped;
    })
    .join('');
}
