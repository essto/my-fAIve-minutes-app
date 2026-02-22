import { Controller, Get, UseGuards, Request } from '@nestjs/common';
import { UserService } from './user.service';

@Controller('users')
export class UserController {
  constructor(private readonly userService: UserService) {}

  @Get('me')
  async getMe(@Request() req: any) {
    // req.user stays populated after JwtGuard
    return req.user;
  }
}
