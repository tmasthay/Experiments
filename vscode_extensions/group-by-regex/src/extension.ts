// import * as vscode from 'vscode';

// export function activate(context: vscode.ExtensionContext) {
//     console.log('Extension "group-by-regex" is now active!');

//     let disposable = vscode.commands.registerCommand('group-by-regex.helloWorld', () => {
//         // Collect all files from visible editors, regardless of file type
//         const visibleEditors = vscode.window.visibleTextEditors;
//         const fileNames = visibleEditors.map(editor => editor.document.fileName);
//         const uniqueFileNames = Array.from(new Set(fileNames)); // Remove any duplicates

//         console.log('Currently loaded files in editors:', uniqueFileNames.join(', '));
//         vscode.window.showInformationMessage(`Loaded files in editors: ${uniqueFileNames.join(', ')}`);

//         // Count and log the number of editor groups
//         const editorGroups = vscode.window.tabGroups.all;
//         console.log('Number of editor groups:', editorGroups.length);
//         vscode.window.showInformationMessage(`Number of editor groups: ${editorGroups.length}`);
//     });

//     context.subscriptions.push(disposable);
// }

// export function deactivate() {}

// import * as vscode from 'vscode';


// export function activate(context: vscode.ExtensionContext) {
//     console.log('Extension "group-by-regex" is now active!');

//     let disposable = vscode.commands.registerCommand('group-by-regex.helloWorld', () => {
//         const editorGroups = vscode.window.tabGroups.all;
//         const fileNames: string[] = []; // Explicit type declaration for fileNames

//         // Iterate over each group and each tab in the group to collect file names
//         editorGroups.forEach(group => {
//             group.tabs.forEach(tab => {
//                 // Check if the input is a file and has a resource property
//                 if ((tab.input as vscode.TabInputText | vscode.TabInputCustom).uri) {
//                     const resource = (tab.input as vscode.TabInputText | vscode.TabInputCustom).uri;
//                     if (resource) {
//                         fileNames.push(resource.fsPath);
//                     }
//                 }
//             });
//         });

//         // Filter out duplicates and .git files
//         const uniqueFileNames = Array.from(new Set(fileNames.filter(name => !name.endsWith('.git'))));

//         console.log('All loaded files across editor groups:', uniqueFileNames.join(', '));
//         vscode.window.showInformationMessage(`Loaded files across editor groups: ${uniqueFileNames.join(', ')}`);

//         // Count and log the number of editor groups
//         console.log('Number of editor groups:', editorGroups.length);
//         vscode.window.showInformationMessage(`Number of editor groups: ${editorGroups.length}`);
//     });

//     context.subscriptions.push(disposable);
// }

// export function deactivate() {}

import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('Extension "group-by-regex" is now active!');

    // Define a map of regex patterns to editor group indexess
	
    const patternMap = new Map<number, RegExp>([
        [1, /\.py$/],    // Group 1 for Python files
        [2, /\.yaml$/],  // Group 2 for YAML files
        [3, /\.gif$/]    // Group 3 for GIF files
    ]);

    let disposable = vscode.commands.registerCommand('group-by-regex.helloWorld', async () => {
        const editorGroups = vscode.window.tabGroups.all;
        const fileNames: string[] = [];

        // Collect all file names from tabs
        editorGroups.forEach(group => {
            group.tabs.forEach(tab => {
                if ((tab.input instanceof vscode.TabInputText) || (tab.input instanceof vscode.TabInputCustom) && tab.input.uri) {
                    fileNames.push(tab.input.uri.fsPath);
                }
            });
        });

        // Filter out duplicates and .git files
        const uniqueFileNames = Array.from(new Set(fileNames.filter(name => !name.endsWith('.git'))));

        // Ensure there are enough editor groups open
        await vscode.commands.executeCommand('workbench.action.closeAllEditors');

        // Ensure there are enough editor groups open
        // while (vscode.window.tabGroups.all.length < patternMap.size + 1) {
        //     await vscode.commands.executeCommand('workbench.action.newGroupRight');
        // }
        await ensureEditorLayout(patternMap.size + 1);

		uniqueFileNames.forEach(async (fileName) => {
			let placed = false;

			// Then open the file in the correct editor group
			patternMap.forEach((regex, groupIndex) => {
				if (regex.test(fileName)) {
					placed = true;
					vscode.commands.executeCommand('vscode.open', vscode.Uri.file(fileName), { viewColumn: groupIndex, preserveFocus: false }).then(() => {
						placed = true;
					});
				}
			});

			// placed = true;
			// If the file does not match any pattern, place it in an additional group
			if (!placed) {
				console.log('Placing file in additional group:', fileName);
				vscode.commands.executeCommand('vscode.open', vscode.Uri.file(fileName), { viewColumn: patternMap.size + 1 });
			}
		});


        console.log('All loaded files across editor groups:', uniqueFileNames.join(', '));
        vscode.window.showInformationMessage(`Loaded files across editor groups: ${uniqueFileNames.join(', ')}`);

        // Log the number of editor groups
        console.log('Number of editor groups:', vscode.window.tabGroups.all.length);
        vscode.window.showInformationMessage(`Number of editor groups: ${vscode.window.tabGroups.all.length}`);
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}

async function ensureEditorLayout(numGroups: number) {
    const layoutConfig = new Map<number, [number, number]>([
        [1, [1, 1]],
        [2, [1, 2]],
        [3, [1, 3]],
        [4, [2, 2]],
        [5, [2, 3]],
        [6, [3, 3]],
        [7, [3, 4]]
    ]);

    const [rows, cols] = layoutConfig.get(numGroups) || [1, numGroups];

    console.log(`Ensuring layout with ${rows} rows and ${cols} columns`)

    // Close all editors to reset the layout
    await vscode.commands.executeCommand('workbench.action.closeAllEditors');

    // Initialize the first group explicitly if not already present
    if (vscode.window.tabGroups.all.length === 0) {
        await vscode.commands.executeCommand('workbench.action.newGroupRight');
    }

    // Create initial row of groups
    for (let col = 1; col < cols; col++) { // Start from 1 because the first group is already created
        await vscode.commands.executeCommand('workbench.action.newGroupRight');
    }

    // Start creating additional rows if necessary
    for (let row = 2; row <= rows; row++) {
        console.log(`Creating row ${row}`);
        for(let col = 1; col < cols; col++){
            // Get us back to the first group of the row
            await vscode.commands.executeCommand('workbench.action.focusLeftGroup');
        }
        await vscode.commands.executeCommand('workbench.action.newGroupBelow');

        // Create remaining groups in the current row
        for (let col = 1; col < cols; col++) {
            console.log(`Creating group ${col + 1} in row ${row}`);
            await vscode.commands.executeCommand('workbench.action.focusRightGroup');
            await vscode.commands.executeCommand('workbench.action.newGroupBelow');
        }
    }
}


