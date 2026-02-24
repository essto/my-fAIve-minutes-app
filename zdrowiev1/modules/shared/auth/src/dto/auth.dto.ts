import { IsEmail, IsString, IsNotEmpty, MinLength, MaxLength, Matches, IsOptional } from 'class-validator';

export class RegisterDto {
    @IsEmail({}, { message: 'Invalid email format' })
    @IsNotEmpty()
    email!: string;

    @IsString()
    @IsNotEmpty()
    @MinLength(8, { message: 'Password must be at least 8 characters long' })
    @MaxLength(64, { message: 'Password is too long' })
    @Matches(/(?=.*[A-Z])/, { message: 'Password must contain at least one uppercase letter' })
    @Matches(/(?=.*[a-z])/, { message: 'Password must contain at least one lowercase letter' })
    @Matches(/(?=.*[0-9])/, { message: 'Password must contain at least one number' })
    password!: string;

    @IsString()
    @IsNotEmpty()
    @MaxLength(100)
    // XSS protection - allow only letters, numbers, spaces and safe punctuation
    @Matches(/^[a-zA-Z0-9\s.,'-]+$/, { message: 'Name contains invalid characters' })
    name!: string;
}

export class LoginDto {
    @IsEmail({}, { message: 'Invalid email format' })
    @IsNotEmpty()
    email!: string;

    @IsString()
    @IsNotEmpty()
    password!: string;
}
