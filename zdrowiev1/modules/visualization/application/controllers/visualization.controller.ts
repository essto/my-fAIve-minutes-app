import { Controller, Get, Post, Body, Req, Query } from '@nestjs/common';
import { VisualizationOrchestrator } from '../../domain/services/visualization-orchestrator.service';

@Controller('visualization')
export class VisualizationController {
  constructor(private readonly orchestrator: VisualizationOrchestrator) {}

  @Get('dashboard')
  async getDashboard(@Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    // In a real app, fetch real data from health/diet modules here
    const mockHealth = { score: 80, heartRate: 72, weightHistory: [], activityData: [] };
    const mockDiet = { score: 75 };
    return this.orchestrator.getDashboardData(userId, mockHealth, mockDiet);
  }

  @Post('export/csv')
  async exportCsv(@Body() body: { data: any[]; headers: string[] }, @Req() req: any) {
    const userId = req.user?.id || '550e8400-e29b-41d4-a716-446655440000';
    return this.orchestrator.exportCsv(userId, body.data, body.headers);
  }
}
