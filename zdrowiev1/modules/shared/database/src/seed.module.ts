import { Module } from '@nestjs/common';
import { AuthModule } from '@monorepo/auth';
import { UserModule } from '@monorepo/user';
import { SeedDemoService } from './seed-demo.service';

@Module({
  imports: [AuthModule, UserModule],
  providers: [SeedDemoService],
  exports: [SeedDemoService],
})
export class SeedModule {}
