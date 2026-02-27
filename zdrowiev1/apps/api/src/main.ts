import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { NestExpressApplication } from '@nestjs/platform-express';
import helmet from 'helmet';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule);

  // Trust proxy is required if you are behind a reverse proxy (e.g. Nginx, AWS ELB)
  // to get the correct user IP for rate limiting.
  app.set('trust proxy', 1);

  // Security Headers (CSP, HSTS, X-Frame-Options, etc.)
  app.use(helmet());

  // CORS — whitelist only known origins
  app.enableCors({
    origin: ['http://localhost:3000', process.env.WEB_URL].filter(Boolean) as string[],
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    credentials: true,
  });

  // Global validation — reject any unvalidated input
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
    }),
  );

  app.setGlobalPrefix('api');
  await app.listen(process.env.PORT || 3006);
}
bootstrap();
