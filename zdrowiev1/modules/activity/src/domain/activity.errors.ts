export class ActivityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ActivityError';
  }
}

export class InvalidActivityDateError extends ActivityError {
  constructor(date: string) {
    super(`Invalid activity date: ${date}`);
    this.name = 'InvalidActivityDateError';
  }
}

export class NegativeStepsError extends ActivityError {
  constructor(steps: number) {
    super(`Steps cannot be negative: ${steps}`);
    this.name = 'NegativeStepsError';
  }
}
