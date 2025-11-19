#!/bin/bash
# Migration helper script

echo "================================"
echo "Python API Migration Helper"
echo "================================"
echo ""
echo "This script helps you switch to the refactored codebase."
echo ""
echo "Current structure:"
echo "  - main.py (old monolithic file, 900+ lines)"
echo ""
echo "New structure:"
echo "  - app/ (modular packages)"
echo "  - main_new.py (new entry point)"
echo ""
echo "Options:"
echo "  1) Backup old main.py and use new structure"
echo "  2) Test new structure (keeps old main.py)"
echo "  3) Revert to old main.py"
echo ""
read -p "Choose option (1-3): " choice

case $choice in
  1)
    echo "Backing up old main.py..."
    mv main.py main_old.py
    mv main_new.py main.py
    echo "✓ Migration complete!"
    echo "✓ Old file saved as main_old.py"
    echo "✓ Run with: python main.py"
    ;;
  2)
    echo "Testing new structure..."
    echo "Run with: python main_new.py"
    echo "(Old main.py is preserved)"
    ;;
  3)
    echo "Reverting to old structure..."
    if [ -f main_old.py ]; then
      mv main_old.py main.py
      echo "✓ Reverted to old main.py"
    else
      echo "⚠ No backup found (main_old.py)"
    fi
    ;;
  *)
    echo "Invalid option"
    exit 1
    ;;
esac

echo ""
echo "Done!"
