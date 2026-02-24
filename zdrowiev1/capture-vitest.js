const cp = require('child_process');
cp.exec('npx vitest run tests/security/rls-isolation.spec.ts --reporter=verbose', (err, stdout, stderr) => {
    console.log("STDOUT\n==============\n", stdout);
    console.log("STDERR\n==============\n", stderr);
    if (err) {
        console.log("EXIT CODE:", err.code);
    }
});
