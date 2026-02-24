const fs = require('fs');
const path = require('path');
const srcDirs = ['apps', 'modules'];
const secretPatterns = [
    /['"]sk[-_][a-zA-Z0-9]{20,}['"]/,   // OpenAI-style
    /['"]ghp_[a-zA-Z0-9]{36,}['"]/,       // GitHub PAT
    /['"]AKIA[A-Z0-9]{16}['"]/,            // AWS Access Key
    /password\s*[:=]\s*['"][^'"]{4,}['"]/i, // password = "value"
];
const violations = [];
function scanDir(dir) {
    if (!fs.existsSync(dir)) return;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            if (!['node_modules', '.git', 'dist', '__tests__', 'test'].includes(entry.name)) {
                scanDir(fullPath);
            }
        } else if (entry.name.endsWith('.ts') && !entry.name.endsWith('.spec.ts') && !entry.name.endsWith('.test.ts')) {
            const content = fs.readFileSync(fullPath, 'utf-8');
            for (const pattern of secretPatterns) {
                if (pattern.test(content)) {
                    violations.push(`${fullPath}: matches ${pattern}`);
                }
            }
        }
    }
}
srcDirs.forEach(scanDir);
console.log("Violations found:", violations);
