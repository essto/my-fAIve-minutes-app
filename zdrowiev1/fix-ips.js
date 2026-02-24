const fs = require('fs');
let content = fs.readFileSync('tests/security/input-validation.spec.ts', 'utf8');
let id = 1;
content = content.replace(/headers: \{ 'Content-Type': 'application\/json' \}/g, () => {
    return `headers: { 'Content-Type': 'application/json', 'X-Forwarded-For': '10.0.0.${id++}' }`;
});
fs.writeFileSync('tests/security/input-validation.spec.ts', content);
console.log('Fixed IPs');
