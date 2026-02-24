'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import * as Sentry from '@sentry/nextjs';

interface Props {
    children?: ReactNode;
}

interface State {
    hasError: boolean;
}

class SentryErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false
    };

    public static getDerivedStateFromError(_: Error): State {
        return { hasError: true };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Uncaught error:', error, errorInfo);
        Sentry.captureException(error);
    }

    public render() {
        if (this.state.hasError) {
            return (
                <div className="p-8 text-center glass-card border-destructive text-destructive m-4">
                    <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
                    <p className="text-sm">Our team has been notified of this issue.</p>
                </div>
            );
        }

        return this.props.children;
    }
}

export default SentryErrorBoundary;
