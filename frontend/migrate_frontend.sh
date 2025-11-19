#!/bin/bash

# Frontend Migration Script
# Switches between monolithic and modular App.tsx

echo "🔧 Frontend Migration Script"
echo "============================"
echo ""

if [ "$1" == "new" ]; then
    echo "📦 Switching to NEW modular structure..."
    if [ -f "src/App.tsx" ]; then
        mv src/App.tsx src/App_old.tsx
        echo "✓ Backed up old App.tsx to App_old.tsx"
    fi
    if [ -f "src/App_new.tsx" ]; then
        cp src/App_new.tsx src/App.tsx
        echo "✓ Activated new modular App.tsx"
    else
        echo "❌ Error: App_new.tsx not found"
        exit 1
    fi
    echo ""
    echo "✅ Migration to modular structure complete!"
    echo "   New structure uses:"
    echo "   - hooks/useImageGeneration.ts"
    echo "   - components/generation/* (9 components)"
    echo "   - components/layout/* (2 components)"
    echo "   - types/generation.ts"
    echo "   - constants/models.ts"
    
elif [ "$1" == "old" ]; then
    echo "📦 Switching to OLD monolithic structure..."
    if [ -f "src/App_old.tsx" ]; then
        cp src/App_old.tsx src/App.tsx
        echo "✓ Restored old monolithic App.tsx"
    else
        echo "❌ Error: App_old.tsx not found"
        exit 1
    fi
    echo ""
    echo "✅ Rollback to monolithic structure complete!"
    
else
    echo "Usage: ./migrate_frontend.sh [new|old]"
    echo ""
    echo "  new  - Switch to new modular structure"
    echo "  old  - Rollback to old monolithic structure"
    echo ""
    echo "Current structure:"
    if [ -f "src/App_new.tsx" ]; then
        echo "  ✓ New modular structure available"
    fi
    if [ -f "src/App_old.tsx" ]; then
        echo "  ✓ Old monolithic structure backed up"
    fi
    if [ -f "src/App.tsx" ]; then
        # Check if it's the new version by looking for imports
        if grep -q "useImageGeneration" src/App.tsx; then
            echo "  → Currently using: NEW modular structure"
        else
            echo "  → Currently using: OLD monolithic structure"
        fi
    fi
fi
