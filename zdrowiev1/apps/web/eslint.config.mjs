import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    rules: {
      "no-restricted-imports": ["error", {
        "paths": [
          {
            "name": "next/navigation",
            "importNames": ["useRouter", "usePathname", "redirect", "Link"],
            "message": "Use imports from '@/i18n/routing' instead — they auto-add locale prefix."
          },
          {
            "name": "next/link",
            "message": "Use { Link } from '@/i18n/routing' instead — it auto-adds locale prefix."
          }
        ]
      }]
    }
  },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
