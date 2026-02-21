import { Module } from '@nestjs/common';
import { VisualizationController } from './application/controllers/visualization.controller';
import { VisualizationOrchestrator } from './domain/services/visualization-orchestrator.service';
import { ChartConfigService } from './domain/services/chart-config.service';
import { DashboardService } from './domain/services/dashboard.service';
import { ExportService } from './domain/services/export.service';

@Module({
  controllers: [VisualizationController],
  providers: [VisualizationOrchestrator, ChartConfigService, DashboardService, ExportService],
  exports: [VisualizationOrchestrator],
})
export class VisualizationModule {}
