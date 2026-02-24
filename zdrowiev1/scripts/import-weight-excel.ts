import { config } from 'dotenv';
import * as XLSX from 'xlsx';
import { drizzle } from 'drizzle-orm/node-postgres';
import { Pool } from 'pg';
import { weightReadings } from '../modules/weight/infrastructure/schemas/weight.schema';
import path from 'path';

config();

const TARGET_USER_ID = process.env.DEFAULT_USER_ID || '550e8400-e29b-41d4-a716-446655440000';

async function main() {
  console.log('Connecting to database...');
  const pool = new Pool({
    host: process.env.POSTGRES_HOST || 'localhost',
    port: parseInt(process.env.POSTGRES_PORT || '5432', 10),
    user: process.env.POSTGRES_USER || 'admin',
    password: process.env.POSTGRES_PASSWORD || 'O6eQx3E2D5qP9S1V0m',
    database: process.env.POSTGRES_DB || 'zdrowiev1',
  });

  const db = drizzle(pool);

  console.log('Ensuring default user exists...');
  await pool.query(
    `INSERT INTO users (id, email, password) VALUES ($1, $2, $3) ON CONFLICT (id) DO NOTHING`,
    [TARGET_USER_ID, `mockuser_${TARGET_USER_ID}@example.com`, 'hashedpassword'],
  );

  console.log('Reading Excel file...');
  const filePath = path.join(__dirname, '../../docs/weight_sebo_interpolated.xlsx');
  const workbook = XLSX.readFile(filePath);
  const sheet = workbook.Sheets[workbook.SheetNames[0]];
  const rows = XLSX.utils.sheet_to_json(sheet) as any[];

  console.log(`Processing ${rows.length} rows...`);

  const records = rows.map((r) => {
    // Some timestamps might be Excel time or string. XLSX often parses DateTime to strings if not configured.
    // If Unnamed: 0 is present
    const dateStr = r['Unnamed: 0'];
    const timestamp = dateStr ? new Date(dateStr) : new Date();

    return {
      userId: TARGET_USER_ID,
      value: Number(r['Waga(Kg)']),
      unit: 'kg' as const,
      bmi: Number(r['BMI']) || null,
      fatPercent: Number(r['Tłuszc(%)']) || null,
      fatKg: Number(r['Masa tłuszczu ciała(Kg)']) || null,
      muscleMassKg: Number(r['Masa mięśni szkieletowych(Kg)']) || null,
      musclePercent: Number(r['Mięśnie(%)']) || null,
      waterPercent: Number(r['Woda(%)']) || null,
      bmrKcal: Number(r['BMR(kcal/dzień)']) || null,
      boneMassKg: Number(r['masa kostna(Kg)']) || null,
      proteinPercent: Number(r['Białka(%)']) || null,
      leanMassKg: Number(r['Waga bez tłuszczu(Kg)']) || null,
      metabolicAge: Number(r['Wiek metaboliczny']) || null,
      timestamp,
      createdAt: new Date(),
      source: 'excel-import',
    };
  });

  // Filter out any invalid records (e.g. empty rows)
  const validRecords = records.filter((r) => !isNaN(r.value));

  console.log(`Inserting ${validRecords.length} valid records in batches of 50...`);

  const chunkSize = 50;
  for (let i = 0; i < validRecords.length; i += chunkSize) {
    const chunk = validRecords.slice(i, i + chunkSize);
    await db.insert(weightReadings).values(chunk);
    process.stdout.write(`Inserted ${i + chunk.length} / ${validRecords.length}\r`);
  }

  console.log('\nImport complete!');
  await pool.end();
}

main().catch((err) => {
  console.error('Import failed:', err);
  process.exit(1);
});
