const fs = require('fs');
const path = require('path');

console.log('--- CI/CD Infrastructure Validator ---');

let hasFailed = false;

function assert(condition, message) {
    if (condition) {
        console.log(`✅ PASS: ${message}`);
    } else {
        console.error(`❌ FAIL: ${message}`);
        hasFailed = true;
    }
}

// 1. Sprawdź istnienie ci.yml
const ciPath = path.join(__dirname, '../.github/workflows/ci.yml');
const ciExists = fs.existsSync(ciPath);
assert(ciExists, '.github/workflows/ci.yml exists');

if (ciExists) {
    const ciContent = fs.readFileSync(ciPath, 'utf8');

    // 2. Sprawdź joby
    assert(ciContent.includes('lint:'), 'ci.yml has lint job');
    assert(ciContent.includes('unit:'), 'ci.yml has unit job');
    assert(ciContent.includes('integration:'), 'ci.yml has integration job');
    assert(ciContent.includes('e2e-web:'), 'ci.yml has e2e-web job');
    assert(ciContent.includes('security:'), 'ci.yml has security job');
    assert(ciContent.includes('build:'), 'ci.yml has build job');

    // 5. Sprawdź wersję Node
    assert(ciContent.includes("node-version: '24'"), 'ci.yml uses node-version 24');

    // 6. Sprawdź przypięte wersje akcji (@v4)
    assert(ciContent.includes('actions/checkout@v4'), 'ci.yml uses actions/checkout@v4');
    assert(!ciContent.includes('actions/checkout@master') && !ciContent.includes('actions/checkout@main') && !ciContent.includes('actions/checkout@v3'), 'ci.yml does not use outdated checkout actions');
}

// 3. Sprawdź istnienie Dockerfiles
const apiDockerPath = path.join(__dirname, '../apps/api/Dockerfile');
assert(fs.existsSync(apiDockerPath), 'apps/api/Dockerfile exists');

const webDockerPath = path.join(__dirname, '../apps/web/Dockerfile');
assert(fs.existsSync(webDockerPath), 'apps/web/Dockerfile exists');

// 4. Sprawdź istnienie .dockerignore
const dockerignorePath = path.join(__dirname, '../.dockerignore');
assert(fs.existsSync(dockerignorePath), '.dockerignore exists');

if (hasFailed) {
    console.error('\n⚠️ Validation Failed! Fix the errors above.');
    process.exit(1);
} else {
    console.log('\n🎉 All checks passed! CI/CD Infrastructure is valid.');
    process.exit(0);
}
