import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Code2, Loader2 } from 'lucide-react';
import { Input } from '@/components/ui/input';

interface SDKAccessPanelProps {
  onModuleSelect: (module: any) => void;
  selectedModuleId?: string;
}

export const SDKAccessPanel = ({ onModuleSelect, selectedModuleId }: SDKAccessPanelProps) => {
  const [searchQuery, setSearchQuery] = useState('');

  const { data: modules, isLoading } = useQuery({
    queryKey: ['sdk-modules'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('modules')
        .select('*')
        .order('created_at', { ascending: false });
      
      if (error) throw error;
      return data;
    },
  });

  const filteredModules = modules?.filter(module =>
    module.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    module.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Card className="glass-effect border-border/50 p-4 h-[calc(100vh-250px)]">
      <div className="mb-4">
        <h2 className="text-lg font-semibold mb-2">Available Modules</h2>
        <Input
          placeholder="Search modules..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="mb-4"
        />
      </div>

      <ScrollArea className="h-[calc(100%-80px)]">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
          </div>
        ) : (
          <div className="space-y-3">
            {filteredModules?.map((module) => (
              <Card
                key={module.id}
                className={`p-3 cursor-pointer transition-all hover:border-primary/50 ${
                  selectedModuleId === module.id ? 'border-primary bg-primary/5' : ''
                }`}
                onClick={() => onModuleSelect({
                  id: module.id,
                  title: module.name,
                  code: module.code || '# No code available',
                  language: module.language || 'python',
                  category: module.module_type,
                  author: 'Developer'
                })}
              >
                <div className="flex items-start gap-2">
                  <Code2 className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-sm truncate">{module.name}</h3>
                    <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
                      {module.description || 'No description'}
                    </p>
                    <div className="flex gap-2 mt-2">
                      <Badge variant="outline" className="text-xs">
                        {module.language}
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        {module.module_type}
                      </Badge>
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </ScrollArea>
    </Card>
  );
};
