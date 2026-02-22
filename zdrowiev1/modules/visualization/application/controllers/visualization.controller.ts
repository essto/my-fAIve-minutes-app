import { Controller, Get, Post, Body, Req } from '@nestjs/common';
import { VisualizationOrchestrator } from '../../domain/services/visualization-orchestrator.service';
import { ExportService } from '../../domain/services/export.service';

@Controller('visualization')
export class VisualizationController {
  constructor(private readonly orchestrator: VisualizationOrchestrator) {}

  @Get('dashboard')
  async getDashboard(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';

    // In a real app, these metrics would come from Health and Diet modules
    const healthMetrics = {
      heartRate: 72,
      bloodPressure: [120, 80] as [number, number],
      oxygenSaturation: 98,
      steps: 8000,
      weightHistory: [],
    };

    const dietMetrics = {
      calories: 2000,
      protein: 100,
      fat: 65,
      carbs: 250,
      water: 2000,
    };

    return this.orchestrator.getDashboardData(userId, healthMetrics, dietMetrics);
  }

  @Post('export/csv')
  async exportCsv(@Body() body: { data: any[]; headers: string[] }) {
    return ExportService.toCSV(body.data, body.headers);
  }
}
